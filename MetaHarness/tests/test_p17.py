from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.llm.chat import LLMError
from metaharness.models import (
    AgentConfig,
    ContextConfig,
    ExecutionRole,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    ProfileDriver,
    SelectionMode,
    UIConfig,
)
from metaharness.orchestrator import Orchestrator
from metaharness.recommendation import (
    ExecutionRecommender,
    RecommendationParseError,
    parse_execution_recommendation,
    render_profile_catalogue,
)
from metaharness.state import RunStateStore


IMPLEMENTER = frozenset({"impl-a", "impl-b"})
REVIEWER = frozenset({"review-a", "review-b"})
VALID = """META EXECUTION RECOMMENDATION v1

IMPLEMENTER_PROFILE: impl-a
REVIEWER_PROFILE: review-a

RATIONALE
Use the least expensive profiles that cover this small change.

END META EXECUTION RECOMMENDATION
"""


def profile(
    profile_id: str,
    role: ExecutionRole,
    *,
    driver: ProfileDriver = ProfileDriver.CODEX,
) -> ModelProfile:
    return ModelProfile(
        id=profile_id,
        display_name=profile_id.title(),
        roles=(role,),
        driver=driver,
        model=f"model-{profile_id}",
        selection_mode=SelectionMode.CLI if driver is ProfileDriver.CODEX else SelectionMode.REQUEST,
        base_url="https://planner.invalid" if driver is ProfileDriver.OPENAI_CHAT else None,
        endpoint_path="/chat" if driver is ProfileDriver.OPENAI_CHAT else None,
        effort="high" if driver is ProfileDriver.CODEX else None,
        sandbox="workspace-write" if driver is ProfileDriver.CODEX else None,
        description="A safe profile.",
        strengths=("implementation",),
    )


class RecommendationParserTests(unittest.TestCase):
    def test_valid_recommendation_and_markdown_rationale_heading(self) -> None:
        value = VALID.replace("RATIONALE", "## RATIONALE")
        result = parse_execution_recommendation(
            value, implementer_ids=IMPLEMENTER, reviewer_ids=REVIEWER
        )
        self.assertEqual(result.implementer_profile, "impl-a")
        self.assertEqual(result.reviewer_profile, "review-a")

    def test_invalid_recommendations_fail_closed(self) -> None:
        cases = {
            "unknown implementer": VALID.replace("impl-a", "impl-x"),
            "unknown reviewer": VALID.replace("review-a", "review-x"),
            "duplicate same implementer": VALID.replace(
                "REVIEWER_PROFILE: review-a", "IMPLEMENTER_PROFILE: impl-a\nREVIEWER_PROFILE: review-a"
            ),
            "duplicate contradictory implementer": VALID.replace(
                "REVIEWER_PROFILE: review-a", "IMPLEMENTER_PROFILE: impl-b\nREVIEWER_PROFILE: review-a"
            ),
            "missing implementer": VALID.replace("IMPLEMENTER_PROFILE: impl-a\n", ""),
            "missing reviewer": VALID.replace("REVIEWER_PROFILE: review-a\n", ""),
            "empty rationale": VALID.replace(
                "Use the least expensive profiles that cover this small change.\n", ""
            ),
            "case mismatch": VALID.replace("impl-a", "IMPL-A"),
            "prose id": VALID.replace("impl-a", "the impl-a profile"),
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(RecommendationParseError):
                    parse_execution_recommendation(
                        value, implementer_ids=IMPLEMENTER, reviewer_ids=REVIEWER
                    )

        oversized = VALID.replace("Use the least expensive profiles that cover this small change.", "x" * 2001)
        with self.assertRaises(RecommendationParseError):
            parse_execution_recommendation(
                oversized, implementer_ids=IMPLEMENTER, reviewer_ids=REVIEWER
            )


class RecommendationTests(unittest.TestCase):
    def test_catalogue_excludes_endpoint_credentials_and_sandbox(self) -> None:
        value = profile("impl-a", ExecutionRole.IMPLEMENTER)
        rendered = render_profile_catalogue([value])
        self.assertIn("ID: impl-a", rendered)
        self.assertIn("EFFORT: high", rendered)
        self.assertNotIn("planner.invalid", rendered)
        self.assertNotIn("workspace-write", rendered)
        self.assertNotIn("api_key_env", rendered)

    def test_recommender_writes_one_request_and_success_artifacts(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def complete(self, prompt: str) -> str:
                self.prompts.append(prompt)
                return VALID

        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            result = ExecutionRecommender(client).recommend(
                "META IMPLEMENTATION CONTRACT v1",
                [profile("impl-a", ExecutionRole.IMPLEMENTER)],
                [profile("review-a", ExecutionRole.REVIEWER, driver=ProfileDriver.OPENAI_CHAT)],
                artifacts_dir=Path(directory),
            )
            self.assertEqual(len(client.prompts), 1)
            self.assertEqual(result.implementer_profile, "impl-a")
            self.assertIn("IMPLEMENTATION CONTRACT", client.prompts[0])
            self.assertNotIn("planner.invalid", client.prompts[0])
            self.assertTrue((Path(directory) / "execution_recommendation.request.txt").exists())
            self.assertTrue((Path(directory) / "execution_recommendation.raw.md").exists())
            self.assertTrue((Path(directory) / "execution_recommendation.json").exists())


class OrchestratorRecommendationTests(unittest.TestCase):
    def _config(self, root: Path, *, enabled: bool = True, multiple: bool = True) -> HarnessConfig:
        planner = profile("planner", ExecutionRole.PLANNER, driver=ProfileDriver.OPENAI_CHAT)
        implementer = profile("impl-a", ExecutionRole.IMPLEMENTER)
        profiles = {planner.id: planner, implementer.id: implementer}
        if multiple:
            profiles["impl-b"] = profile("impl-b", ExecutionRole.IMPLEMENTER)
        profiles["review-a"] = profile("review-a", ExecutionRole.REVIEWER, driver=ProfileDriver.OPENAI_CHAT)
        return HarnessConfig(
            repo=root,
            base_ref="HEAD",
            runs_root=root / "runs",
            worktrees_root=root / "worktrees",
            require_clean_base=True,
            planner=LLMEndpointConfig("https://planner.invalid", "/chat", "planner"),
            reviewer=LLMEndpointConfig("https://reviewer.invalid", "/chat", "reviewer"),
            context=ContextConfig(),
            agent=AgentConfig(),
            check_catalog=(),
            allow_no_required_checks=True,
            ui=UIConfig(
                default_planner_profile="planner",
                default_implementer_profile="impl-a",
                default_reviewer_profile="review-a",
                enable_profile_recommendation=enabled,
            ),
            model_profiles=profiles,
        )

    def test_feature_disabled_and_single_implementer_do_not_call(self) -> None:
        class Client:
            calls = 0

            def complete(self, _prompt: str) -> str:
                self.calls += 1
                return VALID

        for enabled, multiple in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, multiple=multiple), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = root / "run"
                run.mkdir()
                (run / "implementation_contract.md").write_text("contract", encoding="utf-8")
                store = RunStateStore(run / "state.json")
                store.initialize("run")
                client = Client()
                Orchestrator(self._config(root, enabled=enabled, multiple=multiple), recommender_client=client)._maybe_recommend_profiles(store, run, "planner")
                self.assertEqual(client.calls, 0)

    def test_transport_failure_is_fail_open(self) -> None:
        class Client:
            def complete(self, _prompt: str) -> str:
                raise LLMError("transport failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "implementation_contract.md").write_text("contract", encoding="utf-8")
            store = RunStateStore(run / "state.json")
            store.initialize("run")
            Orchestrator(self._config(root), recommender_client=Client())._maybe_recommend_profiles(store, run, "planner")
            state = store.load()
            self.assertEqual(state["status"], "planning")
            self.assertEqual(state["recommendation"]["status"], "FAILED")
            self.assertTrue((run / "execution_recommendation.error.txt").exists())


if __name__ == "__main__":
    unittest.main()
