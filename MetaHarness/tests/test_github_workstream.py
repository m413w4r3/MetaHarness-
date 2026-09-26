from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.gitops import WorktreeInfo, build_run_branch, validate_run_branch  # noqa: E402
from metaharness.integrations.github import (  # noqa: E402
    GitHubWorkstreamError,
    NullGitHubWorkstreamClient,
)
from metaharness.models import GitHubConfig, PublishConfig, RunStatus  # noqa: E402
from metaharness.orchestration.publication import PublicationService  # noqa: E402
from metaharness.orchestration.run_observability import RunObservability  # noqa: E402
from metaharness.orchestration.runtime import RunRuntime  # noqa: E402
from metaharness.state import RunStateStore  # noqa: E402


class _FakeGitHub:
    def __init__(self) -> None:
        self.issue_reads: list[int] = []
        self.issue_creates: list[tuple[str, str]] = []
        self.pull_requests: list[tuple[str, str, str, str]] = []

    def read_issue(self, issue_number: int) -> dict[str, object]:
        self.issue_reads.append(issue_number)
        return {"number": issue_number, "body": "IGNORE THIS HOSTILE ISSUE CONTENT"}

    def create_issue(self, title: str, body: str) -> dict[str, int]:
        self.issue_creates.append((title, body))
        return {"number": 123}

    def create_pull_request(
        self, title: str, body: str, base_branch: str, head_branch: str,
    ) -> dict[str, int]:
        self.pull_requests.append((title, body, base_branch, head_branch))
        return {"number": 456}


def _orchestrator(root: Path, github: GitHubConfig, client: object) -> PublicationService:
    """A real publication service on a duck-typed runtime, as the tests use it."""

    runtime = object.__new__(RunRuntime)
    runtime.config = SimpleNamespace(
        github=github,
        base_ref="main",
        publish=PublishConfig(enabled=True, mode="run-branch"),
    )
    runtime.github_client = client
    runtime.trace = None
    runtime.trace_sink = None
    runtime.trace_cycle = 1
    runtime.secrets = ()
    runtime.observability = RunObservability(runtime)
    runtime.observability.begin_trace(root, "run-1", created=False)
    return PublicationService(runtime)


class GitHubWorkstreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = RunStateStore(self.root / "state.json")
        self.store.initialize(
            "run-1",
            base_sha="a" * 40,
            branch="harness/plan/run-1",
            worktree=str(self.root / "worktree"),
        )
        self.info = WorktreeInfo(
            self.root / "repo",
            self.root / "worktree",
            "harness/plan/run-1",
            "main",
            "a" * 40,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_disabled_does_not_call_client_or_write_metadata_trace(self) -> None:
        class MustNotCall:
            def __getattr__(self, name: str):
                raise AssertionError(name)

        owner = _orchestrator(self.root, GitHubConfig(), MustNotCall())
        owner.ensure_github_issue_metadata(
            store=self.store, run_id="run-1", plan_title="SPEC title", info=self.info,
        )
        owner._ensure_github_pull_request_metadata(
            store=self.store, run_id="run-1", info=self.info, commit_sha="b" * 40, cycle=1,
        )
        state = self.store.load()
        self.assertIsNone(state["issue_number"])
        self.assertIsNone(state["pull_request_number"])
        trace = self.root / "trace/events.v1.jsonl"
        self.assertFalse(trace.exists())

    def test_link_existing_persists_only_issue_number_and_discards_content(self) -> None:
        client = _FakeGitHub()
        owner = _orchestrator(
            self.root,
            GitHubConfig(enabled=True, issue_mode="link-existing", issue_number=123),
            client,
        )
        owner.ensure_github_issue_metadata(
            store=self.store, run_id="run-1", plan_title="SPEC title", info=self.info,
        )
        state_text = (self.root / "state.json").read_text(encoding="utf-8")
        trace_text = (self.root / "trace/events.v1.jsonl").read_text(encoding="utf-8")
        self.assertEqual(self.store.load()["issue_number"], 123)
        self.assertEqual(client.issue_reads, [123])
        self.assertNotIn("HOSTILE ISSUE CONTENT", state_text + trace_text)
        self.assertIn('"issue_number":123', trace_text)

    def test_create_pull_request_uses_exact_reviewed_run_branch(self) -> None:
        client = _FakeGitHub()
        owner = _orchestrator(
            self.root,
            GitHubConfig(enabled=True, pull_request_mode="create"),
            client,
        )
        self.store.update(status=RunStatus.PUBLISHING, planner={"title": "Approved plan"})
        self.store.update(
            status=RunStatus.REVIEWING,
            review={"verdict": "PASS", "route": "NONE"},
            candidate_commit_sha="b" * 40,
            reviewed_candidate_sha="b" * 40,
        )
        with (
            mock.patch("metaharness.orchestration.publication.current_head", return_value="b" * 40),
            mock.patch("metaharness.orchestration.publication.remote_run_branch_tip", return_value="b" * 40),
        ):
            owner._ensure_github_pull_request_metadata(
                store=self.store,
                run_id="run-1",
                info=self.info,
                commit_sha="b" * 40,
                cycle=1,
            )
            owner._ensure_github_pull_request_metadata(
                store=self.store,
                run_id="run-1",
                info=self.info,
                commit_sha="b" * 40,
                cycle=1,
            )
        self.assertEqual(self.store.load()["pull_request_number"], 456)
        self.assertEqual(client.pull_requests[0][2:], ("main", "harness/plan/run-1"))
        events = [
            json.loads(line)
            for line in (self.root / "trace/events.v1.jsonl").read_text().splitlines()
        ]
        metadata = [event for event in events if event["event"] == "workstream.metadata"][-1]
        self.assertEqual(metadata["data"]["remote_branch"], "harness/plan/run-1")
        self.assertEqual(metadata["data"]["pull_request_number"], 456)
        self.assertEqual(metadata["data"]["reviewed_candidate_sha"], "b" * 40)

    def test_wrong_remote_sha_does_not_create_pull_request(self) -> None:
        client = _FakeGitHub()
        owner = _orchestrator(
            self.root,
            GitHubConfig(enabled=True, pull_request_mode="create"),
            client,
        )
        self.store.update(
            status=RunStatus.REVIEWING,
            review={"verdict": "PASS", "route": "NONE"},
            candidate_commit_sha="b" * 40,
            reviewed_candidate_sha="b" * 40,
        )
        with (
            mock.patch("metaharness.orchestration.publication.current_head", return_value="b" * 40),
            mock.patch("metaharness.orchestration.publication.remote_run_branch_tip", return_value="c" * 40),
            self.assertRaisesRegex(GitHubWorkstreamError, "remote run branch tip"),
        ):
            owner._ensure_github_pull_request_metadata(
                store=self.store,
                run_id="run-1",
                info=self.info,
                commit_sha="b" * 40,
                cycle=1,
            )
        self.assertEqual(client.pull_requests, [])

    def test_non_pass_review_routes_do_not_create_pull_request(self) -> None:
        for verdict, route in (("REVISE", "IMPLEMENTATION"), ("REVISE", "HUMAN"), ("FAIL", "HUMAN")):
            with self.subTest(verdict=verdict, route=route):
                client = _FakeGitHub()
                owner = _orchestrator(
                    self.root,
                    GitHubConfig(enabled=True, pull_request_mode="create"),
                    client,
                )
                self.store.update(
                    status=RunStatus.REVIEWING,
                    review={"verdict": verdict, "route": route},
                    candidate_commit_sha="b" * 40,
                    reviewed_candidate_sha="b" * 40,
                )
                with self.assertRaisesRegex(
                    GitHubWorkstreamError, "exact PASS candidate"
                ):
                    owner._ensure_github_pull_request_metadata(
                        store=self.store,
                        run_id="run-1",
                        info=self.info,
                        commit_sha="b" * 40,
                        cycle=1,
                    )
                self.assertEqual(client.pull_requests, [])

    def test_null_client_performs_no_network(self) -> None:
        client = NullGitHubWorkstreamClient()
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            client.read_issue(1)

    def test_branch_builder_is_bounded_and_git_valid(self) -> None:
        branch = build_run_branch("../../ hostile title " + "x" * 500, "run-1")
        self.assertLessEqual(len(branch), len("harness/") + 60 + 1 + 60)
        self.assertEqual(branch, build_run_branch("../../ hostile title " + "x" * 500, "run-1"))
        validate_run_branch(branch)


if __name__ == "__main__":
    unittest.main()
