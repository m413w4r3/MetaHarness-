"""One effective step authority, end to end, across repair, acceptance and resume.

Regression for run 20260923T182533Z-5dec8f9346: two validated contract
repairs authorized two more S05 paths, the worker succeeded inside that
authority, and the commit gate then refused exactly those paths because it
was handed the approved step instead of the effective one.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from metaharness.agent.protocol import CONTRACT_MISMATCH_HEADER
from metaharness.commit_gate import commit_safety_gate
from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestration.implementation import ImplementationService
from metaharness.orchestration.step_authority import mutable_paths, read_step_candidate
from metaharness.orchestrator import Orchestrator
from metaharness.repository_topology import RepositoryTopology
from metaharness.resume import resume_info
from metaharness.run_options import RunOptions
from tests.pipeline.support import repaired_step_contract
from tests.pipeline_support import (
    PipelineHarness, ScriptedChat, check_repair_result, git, plan, review,
)

SPEC = "Make feature.txt good.\n"
EDITION = "frontend/src/features/edition-dashboard/EditionDashboard.test.tsx"
TRANSFER = "frontend/src/components/ProductionStateTransfer.test.tsx"


def mismatch_for(text: str):
    def action(_request) -> str:
        return CONTRACT_MISMATCH_HEADER + "\n" + text
    return action


def writes(*paths: str, content: str = "changed\n", report: str = "done\n"):
    def action(request) -> str:
        for path in paths:
            target = request.worktree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("good\n" if path == "feature.txt" else content, encoding="utf-8")
        return report
    return action


def repair_contract(*paths: str) -> str:
    """The S01 repair: feature.txt plus *paths*, read and written."""

    reads = "".join(f"- {path} :: current content\n" for path in paths)
    written = "".join(f"- {path}\n" for path in paths)
    return repaired_step_contract().replace(
        "- feature.txt :: current content\n", "- feature.txt :: current content\n" + reads, 1,
    ).replace("WRITE_SET\n- feature.txt\n", "WRITE_SET\n- feature.txt\n" + written, 1)


class NoCall:
    """A model client that must not be called."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def complete(self, request: str) -> str:
        self.requests.append(request)
        raise AssertionError("no model call is allowed here")


class StepAuthorityHarness(PipelineHarness):
    TRACKED = ("a.txt", "c.txt", "d.txt", "e.txt", TRANSFER, EDITION)

    def setUp(self) -> None:
        super().setUp()
        for path in self.TRACKED:
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "tracked fixtures")
        git(self.repo, "push", "-q", "origin", "main")

    def options(self, config, *, max_added: int = 4) -> RunOptions:
        return RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=max_added,
        )

    def one_step_plan(self, *extra: str) -> str:
        """S01 approved on feature.txt (A) and *extra* (B...)."""

        raw = plan(("S01", "feature.txt", "Write the feature"))
        reads = "".join(f"- {path} :: current content\n" for path in extra)
        written = "".join(f"- {path}\n" for path in extra)
        return raw.replace(
            "- feature.txt :: current content\n", "- feature.txt :: current content\n" + reads, 1,
        ).replace("WRITE_SET\n- feature.txt\n", "WRITE_SET\n- feature.txt\n" + written, 1)

    def step_dir(self, step_id: str = "S01") -> Path:
        return self.run_dir() / f"cycles/001/implementation/steps/{step_id}"

    def json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def accepted_paths(self, step_id: str = "S01") -> list[str]:
        chain = self.json(self.run_dir() / "accepted-chain.json")["commits"]
        (record,) = [item for item in chain if item["step_id"] == step_id]
        return sorted(record["changed_paths"])

    def run_pipeline(self, planner: list, *, config=None, options=None, reviewer=None):
        config = config or self.config()
        return self.orchestrator(
            config, planner=planner, reviewer=reviewer or [review()],
        ).run_text(SPEC, run_id="run", run_options=options or self.options(config))

    def resume_without_planner(self, reviewer=None):
        planner = NoCall()
        orchestrator = Orchestrator(
            self.config(), planner_client=planner,
            reviewer_client=ScriptedChat(reviewer or [review()], name="reviewer"),
        )
        return orchestrator.resume("run"), planner

    def two_repairs_then_success(self, *success_paths: str) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_for("The step also needs c.txt and d.txt."),
            mismatch_for("The step still needs d.txt."),
            writes(*success_paths),
        )

    def test_check_repair_scope_request_cannot_exceed_the_cycle_envelope(self) -> None:
        scope_request = (
            "META SCOPE REQUEST v1\n\n"
            "REASON\nThe source path is needed to resolve the gate failure.\n\n"
            "PATHS\n- c.txt\n\n"
            "EVIDENCE\n- The failing check depends on c.txt.\n\n"
            "END META SCOPE REQUEST"
        )
        def fail_feature(request):
            (request.worktree / "feature.txt").write_text("bad\n", encoding="utf-8")
            return "implemented with a failing gate\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, fail_feature)
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda _request: scope_request + "\n\n" + check_repair_result(
                "BLOCKED", "NOT_RUN", "SCOPE", "c.txt is outside approved cycle scope",
            ),
        )
        config = self.config(check_repair=1)

        result = self.run_pipeline(
            [self.one_step_plan()], config=config, options=self.options(config, max_added=1),
        )

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_SCOPE_VIOLATION")
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])


class EffectiveAuthorityCommitTests(StepAuthorityHarness):
    def test_two_validated_repairs_authorize_the_commit_gate(self) -> None:
        """CAS 1: A,B approved; repairs add C then D; success on A,B,C,D."""

        self.two_repairs_then_success("feature.txt", "a.txt", "c.txt", "d.txt")
        captured: list[tuple[str, ...]] = []

        def spy(*args, **kwargs):
            captured.append(tuple(kwargs["mutable_scope"]))
            return commit_safety_gate(*args, **kwargs)

        with mock.patch("metaharness.orchestration.implementation.commit_safety_gate", side_effect=spy):
            result = self.run_pipeline([
                self.one_step_plan("a.txt"),
                repair_contract("a.txt", "c.txt"),
                repair_contract("a.txt", "c.txt", "d.txt"),
            ])

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.workers.calls), 3)
        self.assertEqual(
            set(self.workers.calls[-1].mutable_paths), {"feature.txt", "a.txt", "c.txt", "d.txt"},
        )
        # The commit gate received the effective authority, not the approved step.
        self.assertEqual(captured, [("a.txt", "c.txt", "d.txt", "feature.txt")])
        step = self.json(self.step_dir() / "step.json")
        candidate = read_step_candidate(self.step_dir())
        acceptance = self.json(self.step_dir() / "step_acceptance.json")
        self.assertEqual(step["changed_paths"], ["a.txt", "c.txt", "d.txt", "feature.txt"])
        self.assertEqual((step["authority_source"], step["repair_slot"]), ("contract_repair", 2))
        # Worker authority == commit-gate authority == accepted record.
        self.assertEqual(candidate["effective_authority_sha256"], step["effective_authority_sha256"])
        self.assertEqual(acceptance["commit_gate_authority_sha256"], step["effective_authority_sha256"])
        self.assertEqual(
            self.json(self.step_dir() / "step_authority.json")["effective_authority_sha256"],
            step["effective_authority_sha256"],
        )
        (chain,) = self.json(self.run_dir() / "accepted-chain.json")["commits"]
        self.assertEqual(chain["effective_authority_sha256"], step["effective_authority_sha256"])
        self.assertNotEqual(step["effective_contract_sha256"], step["approved_contract_sha256"])
        names = self.trace_names()
        self.assertLess(names.index("step.candidate.persisted"), names.index("step.committed"))

    def test_one_repair_adding_the_two_frontend_fixtures_commits(self) -> None:
        """CAS 2: the exact S05 shape, one repair +2 paths."""

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_for("Fixtures ProductionStateTransfer.test.tsx and EditionDashboard.test.tsx must change."),
            writes("feature.txt", "a.txt", TRANSFER, EDITION),
        )
        result = self.run_pipeline([self.one_step_plan("a.txt"), repair_contract("a.txt", TRANSFER, EDITION)])

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        validation = self.json(self.step_dir() / "contract_repairs/01/validation.json")
        self.assertEqual(validation["added_mutable_paths"], sorted([TRANSFER, EDITION]))
        self.assertEqual(self.accepted_paths(), sorted(["feature.txt", "a.txt", TRANSFER, EDITION]))

    def test_a_path_outside_the_effective_authority_stays_a_violation(self) -> None:
        """CAS 3: effective A..D, worker writes E: never adopted."""

        self.two_repairs_then_success("feature.txt", "c.txt", "d.txt", "e.txt")
        result = self.run_pipeline([
            self.one_step_plan("a.txt"),
            repair_contract("a.txt", "c.txt"),
            repair_contract("a.txt", "c.txt", "d.txt"),
        ])

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_SCOPE_VIOLATION")
        self.assertIn("e.txt", self.state()["failure"]["detail"])
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)
        self.assertFalse((self.step_dir() / "step_candidate.json").exists())

    def test_the_commit_gate_itself_still_refuses_an_unauthorized_path(self) -> None:
        """CAS 3 at the commit boundary: the gate is strict, not permissive."""

        self.workers.on(ExecutionRole.IMPLEMENTER, writes("feature.txt"))
        original = commit_safety_gate

        def narrowed(*args, **kwargs):
            # Simulate a candidate the effective authority does not cover.
            kwargs["mutable_scope"] = ("a.txt",)
            return original(*args, **kwargs)

        with mock.patch("metaharness.orchestration.implementation.commit_safety_gate", side_effect=narrowed):
            result = self.run_pipeline([self.one_step_plan()])

        self.assertEqual(result.status, RunStatus.FAILED)
        failure = self.state()["failure"]
        self.assertEqual(failure["reason"], "COMMIT_GATE_FAILED")
        self.assertIn("COMMIT_SCOPE_VIOLATION", failure["detail"])
        acceptance = self.json(self.step_dir() / "step_acceptance.json")
        self.assertEqual((acceptance["status"], acceptance["code"]), ("refused", "COMMIT_SCOPE_VIOLATION"))
        self.assertEqual(acceptance["paths"], ["feature.txt"])
        info = resume_info(self.run_dir(), self.state())
        self.assertFalse(info.resumable)
        self.assertIsNone(info.operation)

    def test_a_secret_in_an_authorized_path_fails_closed(self) -> None:
        """CAS 13."""

        secret = "sk-test-SUPERSECRETVALUE-0123456789"
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_for("The step also needs c.txt."),
            writes("feature.txt", "c.txt", content=f"token = {secret}\n"),
        )
        with mock.patch("metaharness.orchestrator.config_secret_values", return_value=(secret,)):
            result = self.run_pipeline([self.one_step_plan(), repair_contract("c.txt")])

        self.assertEqual(result.status, RunStatus.FAILED)
        failure = self.state()["failure"]
        self.assertEqual(failure["reason"], "COMMIT_GATE_FAILED")
        self.assertIn("COMMIT_SECURITY_FAILURE", failure["detail"])
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), git(self.repo, "rev-list", "--count", "main"))


class CycleScopeTests(StepAuthorityHarness):
    def test_semantic_revision_receives_the_repaired_step_scope(self) -> None:
        """CAS 11: the cycle authority is the union of effective step authorities."""

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_for("The step also needs c.txt."),
            writes("feature.txt", "c.txt"),
        )
        self.workers.on(ExecutionRole.REVISER, lambda _request: "no semantic change\n")
        config = self.config(semantic_revision=True)
        result = self.run_pipeline([self.one_step_plan(), repair_contract("c.txt")], config=config)

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        reviser = next(call for call in self.workers.calls if call.role is ExecutionRole.REVISER)
        # Exactly the union of accepted effective scopes, nothing more.
        self.assertEqual(set(reviser.mutable_paths), {"feature.txt", "c.txt"})


class TopologyEvidenceTests(StepAuthorityHarness):
    def test_the_repair_prompt_lists_the_tracked_candidate_of_a_basename(self) -> None:
        """CAS 4: the mismatch names only a basename; the planner still chooses."""

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_for("EditionDashboard.test.tsx asserts the removed link."),
            writes("feature.txt", EDITION),
        )
        result = self.run_pipeline([self.one_step_plan(), repair_contract(EDITION)])

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        request = self.planner.requests[1]
        self.assertIn(
            "<REPOSITORY PATH CANDIDATES>\nEditionDashboard.test.tsx:\n  - " + EDITION
            + "\n</REPOSITORY PATH CANDIDATES>",
            request,
        )
        evidence = self.json(self.step_dir() / "contract_repairs/01/topology_evidence.json")
        self.assertEqual(evidence["tree_sha"], self.json(self.step_dir() / "contract_repairs/01/transaction.json")["tree_sha"])
        self.assertEqual(evidence["references"][0]["candidates"], [EDITION])

    def test_a_wrong_directory_is_corrected_with_the_exact_tracked_candidate(self) -> None:
        """CAS 5: rejected, corrected in the same slot, worker not replayed."""

        wrong = "frontend/src/features/edition-workflow/EditionDashboard.test.tsx"
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_for("EditionDashboard.test.tsx asserts the removed link."),
            writes("feature.txt", EDITION),
        )
        result = self.run_pipeline([self.one_step_plan(), repair_contract(wrong), repair_contract(EDITION)])

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.workers.calls), 2)
        slot = self.step_dir() / "contract_repairs/01"
        self.assertEqual(sorted(path.name for path in slot.parent.iterdir()), ["01"])
        correction = self.planner.requests[2]
        self.assertIn(
            f"INVALID PATH:\n  {wrong}\n\nTRACKED CANDIDATES FOR BASENAME:\n  - {EDITION}",
            correction,
        )
        self.assertTrue((slot / "output_attempts/002/topology_evidence.json").is_file())
        self.assertEqual(self.json(slot / "validation.json")["added_mutable_paths"], [EDITION])
        self.assertEqual(self.json(slot / "transaction.json")["status"], "completed")

    def test_an_ambiguous_basename_lists_every_candidate_and_chooses_none(self) -> None:
        """CAS 6."""

        topology = RepositoryTopology("a" * 40, frozenset({
            "one/Widget.test.tsx", "two/Widget.test.tsx", "three/Other.tsx",
        }))
        (entry,) = topology.evidence("Widget.test.tsx fails")
        self.assertEqual(entry["candidates"], ["one/Widget.test.tsx", "two/Widget.test.tsx"])
        self.assertEqual(topology.candidates("three/Other.tsx"), ())
        self.assertEqual(topology.candidates("zzz/Other.tsx"), ("three/Other.tsx",))


class StepAcceptanceResumeTests(StepAuthorityHarness):
    def interrupt_before_commit(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_for("The step also needs c.txt."),
            writes("feature.txt", "c.txt"),
        )
        with mock.patch(
            "metaharness.orchestration.implementation.commit_safety_gate", side_effect=KeyboardInterrupt(),
        ):
            result = self.run_pipeline([self.one_step_plan(), repair_contract("c.txt")])
        self.assertEqual(result.status, RunStatus.INTERRUPTED)
        self.assertEqual(self.checkpoint()["phase"], "step_acceptance")
        self.assertIsNotNone(read_step_candidate(self.step_dir()))

    def test_a_crash_after_worker_success_resumes_without_any_model_call(self) -> None:
        """CAS 7."""

        self.interrupt_before_commit()
        info = resume_info(self.run_dir(), self.state())
        self.assertTrue(info.resumable)
        self.assertEqual((info.label, info.operation), ("Retry step acceptance (S01)", "step_acceptance"))
        calls = len(self.workers.calls)

        resumed, planner = self.resume_without_planner()

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.workers.calls), calls)
        self.assertEqual(planner.requests, [])
        self.assertEqual(self.accepted_paths(), ["c.txt", "feature.txt"])
        self.assertIn("recovery.resumed", self.trace_names())

    def test_a_crash_after_the_commit_is_only_recorded_again(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_for("The step also needs c.txt."),
            writes("feature.txt", "c.txt"),
        )
        with mock.patch.object(
            ImplementationService, "_finalize_accepted_step", side_effect=KeyboardInterrupt(),
        ):
            result = self.run_pipeline([self.one_step_plan(), repair_contract("c.txt")])
        self.assertEqual(result.status, RunStatus.INTERRUPTED)
        committed = git(self.worktree(), "rev-parse", "HEAD")

        resumed, planner = self.resume_without_planner()

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(planner.requests, [])
        (chain,) = self.json(self.run_dir() / "accepted-chain.json")["commits"]
        self.assertEqual(chain["commit_sha"], committed)
        self.assertEqual(git(self.worktree(), "rev-list", "--count", f"{self.state()['base_sha']}..{committed}"), "1")

    def test_a_missing_or_corrupt_candidate_fails_closed(self) -> None:
        """CAS 8."""

        self.interrupt_before_commit()
        path = self.step_dir() / "step_candidate.json"
        payload = self.json(path)
        payload["changed_paths"] = ["feature.txt"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        calls = len(self.workers.calls)

        resumed, planner = self.resume_without_planner()

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertIn("step candidate hash changed", self.state()["failure"]["detail"])
        self.assertEqual((len(self.workers.calls), planner.requests), (calls, []))

    def test_a_corrupted_effective_authority_fails_closed(self) -> None:
        """CAS 12: contract.md, then validation.json, tampered."""

        for artifact in ("contract.md", "validation.json"):
            with self.subTest(artifact=artifact):
                self.tearDown()
                self.setUp()
                self.interrupt_before_commit()
                target = self.step_dir() / "contract_repairs/01" / artifact
                if artifact == "contract.md":
                    target.write_text(target.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
                else:
                    payload = self.json(target)
                    payload["added_mutable_paths"] = ["c.txt", "e.txt"]
                    target.write_text(json.dumps(payload), encoding="utf-8")

                resumed, planner = self.resume_without_planner()

                self.assertEqual(resumed.status, RunStatus.FAILED)
                self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
                self.assertEqual(planner.requests, [])
                self.assertEqual(len(self.workers.calls), 2)

    def test_git_ownership_drift_fails_closed(self) -> None:
        """CAS 14."""

        self.interrupt_before_commit()
        git(self.worktree(), "checkout", "-q", "-b", "operator-branch")

        resumed, planner = self.resume_without_planner()

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(planner.requests, [])


class StagedReliabilityTests(StepAuthorityHarness):
    """Part O: six steps, S05 repaired +2 paths, gates, revision, review."""

    STEP_FILES = ("s01.txt", "s02.txt", "s03.txt", "s04.txt")

    def setUp(self) -> None:
        super().setUp()
        for path in self.STEP_FILES:
            (self.repo / path).write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "step files")
        git(self.repo, "push", "-q", "origin", "main")

    def test_the_repaired_s05_is_accepted_and_the_run_commits(self) -> None:
        steps = (
            ("S01", "s01.txt", "First stage"), ("S02", "s02.txt", "Second stage"),
            ("S03", "s03.txt", "Third stage"), ("S04", "s04.txt", "Fourth stage"),
            ("S05", "a.txt", "Write the feature"), ("S06", "feature.txt", "Finish"),
        )
        repaired = repair_contract(TRANSFER, EDITION).replace("feature.txt", "a.txt").replace(
            "STEP_ID: S01", "STEP_ID: S05", 1,
        )
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            *(writes(path) for path in self.STEP_FILES),
            mismatch_for("ProductionStateTransfer.test.tsx and EditionDashboard.test.tsx must change."),
            writes("a.txt", TRANSFER, EDITION),
            writes("feature.txt"),
        )
        self.workers.on(ExecutionRole.REVISER, lambda _request: "no semantic change\n")
        config = self.config(semantic_revision=True)

        result = self.run_pipeline([plan(*steps), repaired], config=config)

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        roles = self.workers.roles()
        self.assertEqual(roles.count("implementer"), 7)  # S01-S04, S05 x2, S06
        self.assertEqual(
            [call.artifact_dir.name for call in self.workers.calls if call.role is ExecutionRole.IMPLEMENTER],
            ["S01", "S02", "S03", "S04", "S05", "S05", "S06"],
        )
        chain = self.json(self.run_dir() / "accepted-chain.json")["commits"]
        self.assertEqual([item["step_id"] for item in chain], ["S01", "S02", "S03", "S04", "S05", "S06"])
        s05 = chain[4]
        self.assertEqual(sorted(s05["changed_paths"]), sorted(["a.txt", TRANSFER, EDITION]))
        self.assertEqual((s05["authority_source"], s05["repair_slot"]), ("contract_repair", 1))
        self.assertTrue(all(item["effective_authority_sha256"] for item in chain))
        self.assertEqual(
            [item["authority_source"] for item in chain].count("approved"), 5,
        )
        repairs = [
            item for item in self.state().get("recovery_attempts", [])
            if item.get("budget_key") == "contract_repairs"
        ]
        self.assertEqual(len(repairs), 1)
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(len(self.reviewer.requests), 1)
