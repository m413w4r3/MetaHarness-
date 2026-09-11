"""Execution profiles: config, immutable selection, approval binding, runtime."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness import approval as approval_module  # noqa: E402
from metaharness.agent.base import AgentResult  # noqa: E402
from metaharness.approval import (  # noqa: E402
    ApprovalDecision,
    PlanIdentity,
    compute_plan_identity,
    compute_plan_identity_from_run,
    read_plan_approval,
    write_plan_approval,
)
from metaharness.cli import main  # noqa: E402
from metaharness.config import ConfigError, load_config  # noqa: E402
from metaharness.execution_selection import (  # noqa: E402
    ExecutionSelectionConflict,
    ExecutionSelectionError,
    ensure_execution_selection,
    is_profile_aware_run,
    read_execution_selection,
    resolve_execution_selection,
    validate_execution_selection,
)
from metaharness.models import (  # noqa: E402
    AgentConfig,
    ApprovalConfig,
    ContextConfig,
    ExecutionRole,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    ProfileDriver,
    RunStatus,
    SelectionMode,
    UIConfig,
)
from metaharness.orchestrator import Orchestrator  # noqa: E402
from metaharness.profiles import (  # noqa: E402
    profile_execution_fingerprint,
    safe_profile_metadata,
)
from metaharness.state import RunStateStore  # noqa: E402
from metaharness.web import api  # noqa: E402
from metaharness.web.api import WebAPIError, approve_run, create_run  # noqa: E402
from metaharness.web.run_manager import (  # noqa: E402
    RunCapacityError,
    RunCollisionError,
    RunManager,
    RunManagerError,
)


PLAN = """STATUS: READY
TITLE: Add the feature
OBJECTIVE: Implement the requested feature.
CONSTRAINTS: Keep the change local.
FILES: feature.txt
IMPLEMENTATION: Create feature.txt with the requested content.
ACCEPTANCE: The feature file exists.
TESTS: Run the configured test.
RISKS: NONE
BLOCKERS: NONE
"""

PASS_REVIEW = """VERDICT: PASS
ROUTE: NONE
SUMMARY: The implementation is acceptable.
FINDINGS: NONE
REQUIRED FIXES: NONE
MISSING TESTS: NONE
RESIDUAL RISKS: NONE
"""

ENV_ALLOWLIST = ("PATH", "HOME", "LANG")


def openai_profile(profile_id: str, roles: tuple[ExecutionRole, ...], **overrides: Any) -> ModelProfile:
    values: dict[str, Any] = dict(
        id=profile_id,
        display_name=profile_id.title(),
        roles=roles,
        driver=ProfileDriver.OPENAI_CHAT,
        model=f"model-{profile_id}",
        selection_mode=SelectionMode.REQUEST,
        base_url="https://llm.invalid",
        endpoint_path=f"/{profile_id}",
        api_key_env="META_PROFILE_TEST_KEY",
        timeout_seconds=60,
        retries=0,
        extra_body={"new_chat": True},
        description="An OpenAI-compatible profile.",
        strengths=("review",),
    )
    values.update(overrides)
    return ModelProfile(**values)


def codex_profile(profile_id: str, **overrides: Any) -> ModelProfile:
    values: dict[str, Any] = dict(
        id=profile_id,
        display_name=profile_id.title(),
        roles=(ExecutionRole.IMPLEMENTER,),
        driver=ProfileDriver.CODEX,
        model=f"codex-{profile_id}",
        selection_mode=SelectionMode.CLI,
        effort="high",
        sandbox="workspace-write",
        timeout_seconds=30,
        description="A Codex profile.",
        strengths=("implementation",),
    )
    values.update(overrides)
    return ModelProfile(**values)


def default_profiles() -> dict[str, ModelProfile]:
    profiles = [
        openai_profile("planner", (ExecutionRole.PLANNER,)),
        codex_profile("impl-a", effort="low"),
        codex_profile("impl-b", effort="high"),
        openai_profile("review-a", (ExecutionRole.REVIEWER,)),
        openai_profile("review-b", (ExecutionRole.REVIEWER,)),
    ]
    return {profile.id: profile for profile in profiles}


def make_config(
    root: Path,
    profiles: dict[str, ModelProfile] | None = None,
    *,
    env_allowlist: tuple[str, ...] = ENV_ALLOWLIST,
    require_plan_approval: bool = True,
    max_active_runs: int = 1,
) -> HarnessConfig:
    endpoint = LLMEndpointConfig("https://llm.invalid", "/planner", "model-planner")
    return HarnessConfig(
        repo=root / "repo",
        base_ref="HEAD",
        runs_root=root / "runs",
        worktrees_root=root / "worktrees",
        require_clean_base=True,
        planner=endpoint,
        reviewer=endpoint,
        context=ContextConfig(always_files=()),
        agent=AgentConfig(env_allowlist=env_allowlist),
        checks=(),
        allow_no_required_checks=True,
        approval=ApprovalConfig(
            require_plan_approval=require_plan_approval, poll_interval_seconds=0.01
        ),
        ui=UIConfig(
            max_active_runs=max_active_runs,
            default_planner_profile="planner",
            default_implementer_profile="impl-a",
            default_reviewer_profile="review-a",
            enable_profile_recommendation=False,
        ),
        model_profiles=profiles if profiles is not None else default_profiles(),
    )


def replace_profile(config: HarnessConfig, profile_id: str, **changes: Any) -> HarnessConfig:
    profiles = dict(config.model_profiles)
    profiles[profile_id] = dataclasses.replace(profiles[profile_id], **changes)
    return dataclasses.replace(config, model_profiles=profiles)


def selection_for(
    config: HarnessConfig, implementer: str = "impl-a", reviewer: str = "review-a"
):
    return resolve_execution_selection(
        config,
        planner_profile_id="planner",
        implementer_profile_id=implementer,
        reviewer_profile_id=reviewer,
    )


def awaiting_run(runs: Path, run_id: str, *, profile_aware: bool = True) -> Path:
    """A run stopped at the plan gate, as written by the orchestrator."""

    run_dir = runs / run_id
    store = RunStateStore(run_dir / "state.json")
    store.initialize(run_id)
    raw, contract = PLAN, "# contract\n"
    (run_dir / "planner.raw.md").write_text(raw, encoding="utf-8")
    (run_dir / "implementation_contract.md").write_text(contract, encoding="utf-8")
    fields: dict[str, Any] = {
        "plan_identity": dataclasses.asdict(compute_plan_identity(raw, contract))
    }
    if profile_aware:
        fields["execution"] = {
            "planner": {
                "profile_id": "planner",
                "model": "model-planner",
                "selection_mode": "request",
            }
        }
    store.update(status=RunStatus.AWAITING_PLAN_APPROVAL, **fields)
    return run_dir


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------- config


PROFILE_TOML = """\
repo = "{root}/repo"
base_ref = "HEAD"
runs_root = "{root}/runs"
worktrees_root = "{root}/worktrees"
allow_no_required_checks = true

[ui]
default_planner_profile = "chat"
default_implementer_profile = "impl"
default_reviewer_profile = "chat"

[model_profiles.chat]
display_name = "Chat"
roles = ["planner", "reviewer"]
driver = "openai-chat"
model = "chat-model"
selection_mode = "request"
base_url = "https://chat.invalid"
endpoint_path = "/v1/chat"
api_key_env = "META_PROFILE_TEST_KEY"
timeout_seconds = 60
retries = 1
{chat_extra}
[model_profiles.impl]
display_name = "Impl"
roles = ["implementer"]
driver = "codex"
model = "codex-model"
selection_mode = "cli"
effort = "high"
sandbox = "workspace-write"
timeout_seconds = 600
{impl_extra}
[model_profiles.chat.extra_body]
new_chat = true
"""


class ProfileConfigTests(unittest.TestCase):
    def load(self, *, chat_extra: str = "", impl_extra: str = "", replace: tuple[str, str] | None = None) -> HarnessConfig:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        content = PROFILE_TOML.format(root=directory, chat_extra=chat_extra, impl_extra=impl_extra)
        if replace is not None:
            self.assertIn(replace[0], content)
            content = content.replace(*replace)
        path = directory / "config.toml"
        path.write_text(content, encoding="utf-8")
        return load_config(path)

    def test_explicit_openai_profile_is_valid(self) -> None:
        profile = self.load().model_profiles["chat"]
        self.assertIs(profile.driver, ProfileDriver.OPENAI_CHAT)
        self.assertEqual(profile.roles, (ExecutionRole.PLANNER, ExecutionRole.REVIEWER))
        self.assertEqual(profile.base_url, "https://chat.invalid")
        self.assertEqual(profile.api_key_env, "META_PROFILE_TEST_KEY")
        self.assertEqual((profile.timeout_seconds, profile.retries), (60, 1))
        self.assertEqual(profile.extra_body, {"new_chat": True})

    def test_explicit_codex_profile_is_valid(self) -> None:
        profile = self.load().model_profiles["impl"]
        self.assertIs(profile.driver, ProfileDriver.CODEX)
        self.assertIs(profile.selection_mode, SelectionMode.CLI)
        self.assertEqual((profile.model, profile.effort, profile.sandbox), ("codex-model", "high", "workspace-write"))
        self.assertEqual(profile.timeout_seconds, 600)

    def test_unknown_profile_keys_are_rejected(self) -> None:
        for name, kwargs in {
            "timeuot_seconds": {"chat_extra": "timeuot_seconds = 5"},
            "modle": {"chat_extra": 'modle = "x"'},
            "effrot": {"impl_extra": 'effrot = "low"'},
            "base_url": {"impl_extra": 'base_url = "https://x.invalid"'},
            "effort": {"chat_extra": 'effort = "high"'},
        }.items():
            with self.subTest(key=name):
                with self.assertRaisesRegex(ConfigError, rf"\.{name} is not allowed"):
                    self.load(**kwargs)

    def test_wrong_driver_or_role_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "driver/role mismatch"):
            self.load(replace=('roles = ["implementer"]', 'roles = ["reviewer"]'))
        with self.assertRaisesRegex(ConfigError, "driver/role mismatch"):
            self.load(replace=('roles = ["planner", "reviewer"]', 'roles = ["planner", "implementer"]'))
        with self.assertRaisesRegex(ConfigError, "driver is invalid"):
            self.load(replace=('driver = "codex"', 'driver = "claude"'))

    def test_codex_retries_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, r"impl\.retries is not allowed for codex"):
            self.load(impl_extra="retries = 2")

    def test_selection_modes_per_driver(self) -> None:
        external = self.load(replace=('selection_mode = "request"', 'selection_mode = "external-ui"'))
        self.assertIs(external.model_profiles["chat"].selection_mode, SelectionMode.EXTERNAL_UI)
        with self.assertRaisesRegex(ConfigError, "must not be cli"):
            self.load(replace=('selection_mode = "request"', 'selection_mode = "cli"'))
        with self.assertRaisesRegex(ConfigError, "must be cli"):
            self.load(replace=('selection_mode = "cli"', 'selection_mode = "request"'))

    def test_safe_profile_metadata_has_no_endpoint_or_credentials(self) -> None:
        metadata = safe_profile_metadata(self.load().model_profiles["chat"])
        for key in ("base_url", "endpoint_path", "api_key_env", "extra_body"):
            self.assertNotIn(key, metadata)
        rendered = json.dumps(metadata)
        for value in ("chat.invalid", "/v1/chat", "META_PROFILE_TEST_KEY", "new_chat"):
            self.assertNotIn(value, rendered)


# ------------------------------------------------------- execution selection


class ExecutionSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = make_config(self.root)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()

    def test_is_profile_aware_run_uses_only_planner_profile_id(self) -> None:
        self.assertTrue(is_profile_aware_run({"execution": {"planner": {"profile_id": "p"}}}))
        for state in (
            {},
            {"execution": {}},
            {"execution": {"planner": {}}},
            {"execution": {"planner": {"profile_id": ""}}},
            {"execution": {"planner": {"profile_id": 3}}},
            {"execution": None},
            {"planner": {"profile_id": "p"}, "execution": {"implementer": {"profile_id": "x"}}},
        ):
            with self.subTest(state=state):
                self.assertFalse(is_profile_aware_run(state))

    def test_new_selection_writes_schema_2_with_three_fingerprints(self) -> None:
        ensure_execution_selection(self.run_dir, selection_for(self.config))
        payload = json.loads((self.run_dir / "execution_selection.json").read_text())
        self.assertEqual(payload["schema_version"], 2)
        for name in ("planner", "implementer", "reviewer"):
            with self.subTest(role=name):
                self.assertRegex(payload[name]["config_sha256"], r"\A[0-9a-f]{64}\Z")
        self.assertEqual(read_execution_selection(self.run_dir), selection_for(self.config))

    def test_snapshot_contains_no_secret_value(self) -> None:
        secret = "sk-profile-secret-value-0042"
        with mock.patch.dict(os.environ, {"META_PROFILE_TEST_KEY": secret}):
            ensure_execution_selection(self.run_dir, selection_for(self.config))
        content = (self.run_dir / "execution_selection.json").read_text()
        self.assertNotIn(secret, content)
        self.assertNotIn("llm.invalid", content)

    def test_schema_1_historic_file_remains_readable(self) -> None:
        historic = {
            "schema_version": 1,
            **{
                name: {
                    "profile_id": profile_id,
                    "driver": profile.driver.value,
                    "model": profile.model,
                    "selection_mode": profile.selection_mode.value,
                    "effort": profile.effort,
                    "sandbox": profile.sandbox,
                }
                for name, profile_id, profile in (
                    ("planner", "planner", self.config.model_profiles["planner"]),
                    ("implementer", "impl-a", self.config.model_profiles["impl-a"]),
                    ("reviewer", "review-a", self.config.model_profiles["review-a"]),
                )
            },
        }
        (self.run_dir / "execution_selection.json").write_text(json.dumps(historic))
        selection = read_execution_selection(self.run_dir)
        self.assertEqual(selection.schema_version, 1)
        self.assertIsNone(selection.implementer.config_sha256)
        validate_execution_selection(self.config, selection)

    def test_schema_2_requires_every_fingerprint(self) -> None:
        ensure_execution_selection(self.run_dir, selection_for(self.config))
        path = self.run_dir / "execution_selection.json"
        payload = json.loads(path.read_text())
        for mutation in ("missing", "null", "bad"):
            broken = json.loads(json.dumps(payload))
            if mutation == "missing":
                del broken["reviewer"]["config_sha256"]
            else:
                broken["reviewer"]["config_sha256"] = None if mutation == "null" else "abc"
            path.unlink()
            path.write_text(json.dumps(broken))
            with self.subTest(mutation=mutation), self.assertRaises(ExecutionSelectionError):
                read_execution_selection(self.run_dir)
        incomplete = dataclasses.replace(
            selection_for(self.config),
            planner=dataclasses.replace(selection_for(self.config).planner, config_sha256=None),
        )
        with self.assertRaises(ExecutionSelectionError):
            validate_execution_selection(self.config, incomplete)

    def test_same_selection_ensure_is_idempotent(self) -> None:
        selection = selection_for(self.config)
        ensure_execution_selection(self.run_dir, selection)
        before = (self.run_dir / "execution_selection.json").read_bytes()
        self.assertEqual(ensure_execution_selection(self.run_dir, selection), selection)
        self.assertEqual((self.run_dir / "execution_selection.json").read_bytes(), before)

    def test_different_selection_cannot_overwrite(self) -> None:
        ensure_execution_selection(self.run_dir, selection_for(self.config, "impl-a"))
        before = (self.run_dir / "execution_selection.json").read_bytes()
        with self.assertRaises(ExecutionSelectionConflict):
            ensure_execution_selection(self.run_dir, selection_for(self.config, "impl-b"))
        self.assertEqual((self.run_dir / "execution_selection.json").read_bytes(), before)
        self.assertEqual([p.name for p in self.run_dir.iterdir()], ["execution_selection.json"])

    def test_concurrent_ensure_publishes_exactly_one_selection(self) -> None:
        for attempt in range(10):
            run_dir = self.root / f"race-{attempt}"
            run_dir.mkdir()
            barrier = threading.Barrier(2)
            outcomes: dict[str, str] = {}

            def claim(implementer: str) -> None:
                barrier.wait()
                try:
                    ensure_execution_selection(run_dir, selection_for(self.config, implementer))
                    outcomes[implementer] = "ok"
                except ExecutionSelectionConflict:
                    outcomes[implementer] = "conflict"

            threads = [threading.Thread(target=claim, args=(i,)) for i in ("impl-a", "impl-b")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertEqual(sorted(outcomes.values()), ["conflict", "ok"])
            winner = next(i for i, outcome in outcomes.items() if outcome == "ok")
            self.assertEqual(read_execution_selection(run_dir).implementer.profile_id, winner)

    def test_fingerprint_changes_for_execution_fields(self) -> None:
        chat = self.config.model_profiles["review-a"]
        codex = self.config.model_profiles["impl-a"]
        base_chat = profile_execution_fingerprint(chat)
        base_codex = profile_execution_fingerprint(codex, agent_env_allowlist=ENV_ALLOWLIST)
        for name, changes in {
            "model": {"model": "other"},
            "driver": {"driver": ProfileDriver.CODEX},
            "selection_mode": {"selection_mode": SelectionMode.EXTERNAL_UI},
            "base_url": {"base_url": "https://other.invalid"},
            "endpoint_path": {"endpoint_path": "/other"},
            "api_key_env": {"api_key_env": "OTHER_KEY"},
            "timeout": {"timeout_seconds": 61},
            "retries": {"retries": 3},
            "extra_body": {"extra_body": {"new_chat": False}},
        }.items():
            with self.subTest(field=name):
                changed = dataclasses.replace(chat, **changes)
                self.assertNotEqual(profile_execution_fingerprint(changed), base_chat)
        for name, changes in {
            "model": {"model": "other"},
            "effort": {"effort": "medium"},
            "sandbox": {"sandbox": "read-only"},
            "timeout": {"timeout_seconds": 31},
        }.items():
            with self.subTest(field=name):
                changed = dataclasses.replace(codex, **changes)
                self.assertNotEqual(
                    profile_execution_fingerprint(changed, agent_env_allowlist=ENV_ALLOWLIST),
                    base_codex,
                )
        self.assertNotEqual(
            profile_execution_fingerprint(codex, agent_env_allowlist=ENV_ALLOWLIST + ("EXTRA",)),
            base_codex,
        )

    def test_fingerprint_ignores_advisory_fields(self) -> None:
        advisory = {
            "description": "Something else.",
            "cost_tier": "high",
            "latency_tier": "slow",
            "strengths": ("other",),
            "display_name": "Renamed",
        }
        for profile_id in ("review-a", "impl-a"):
            profile = self.config.model_profiles[profile_id]
            for name, value in advisory.items():
                with self.subTest(profile=profile_id, field=name):
                    changed = dataclasses.replace(profile, **{name: value})
                    self.assertEqual(
                        profile_execution_fingerprint(changed, agent_env_allowlist=ENV_ALLOWLIST),
                        profile_execution_fingerprint(profile, agent_env_allowlist=ENV_ALLOWLIST),
                    )

    def test_validation_detects_every_execution_change(self) -> None:
        selection = selection_for(self.config, "impl-b", "review-b")
        validate_execution_selection(self.config, selection)
        cases: dict[str, HarnessConfig] = {
            "implementer model": replace_profile(self.config, "impl-b", model="changed"),
            "implementer effort": replace_profile(self.config, "impl-b", effort="low"),
            "implementer sandbox": replace_profile(self.config, "impl-b", sandbox="read-only"),
            "implementer timeout": replace_profile(self.config, "impl-b", timeout_seconds=99),
            "reviewer driver": replace_profile(self.config, "review-b", driver=ProfileDriver.CODEX),
            "reviewer selection_mode": replace_profile(
                self.config, "review-b", selection_mode=SelectionMode.EXTERNAL_UI
            ),
            "reviewer endpoint": replace_profile(self.config, "review-b", endpoint_path="/moved"),
            "reviewer api_key_env": replace_profile(self.config, "review-b", api_key_env="OTHER"),
            "reviewer retries": replace_profile(self.config, "review-b", retries=5),
            "reviewer extra_body": replace_profile(self.config, "review-b", extra_body={"x": 1}),
            "planner model": replace_profile(self.config, "planner", model="changed"),
            "env allowlist": dataclasses.replace(
                self.config, agent=AgentConfig(env_allowlist=ENV_ALLOWLIST + ("SECRET_TOKEN",))
            ),
            "profile removed": dataclasses.replace(
                self.config,
                model_profiles={
                    k: v for k, v in self.config.model_profiles.items() if k != "impl-b"
                },
            ),
        }
        for name, changed in cases.items():
            with self.subTest(case=name), self.assertRaises(ExecutionSelectionError):
                validate_execution_selection(changed, selection)
        for field, value in (
            ("description", "Other."),
            ("strengths", ("x",)),
            ("cost_tier", "low"),
            ("latency_tier", "fast"),
            ("display_name", "Renamed"),
        ):
            with self.subTest(advisory=field):
                validate_execution_selection(
                    replace_profile(self.config, "impl-b", **{field: value}), selection
                )


# ------------------------------------------------------------------ approval


class ApprovalBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = make_config(self.root)
        self.runs = self.config.runs_root
        self.runs.mkdir()

    def approve(self, run_id: str, implementer: str = "impl-b", reviewer: str = "review-b") -> dict:
        return approve_run(
            self.runs,
            run_id,
            "APPROVE",
            config=self.config,
            implementer_profile=implementer,
            reviewer_profile=reviewer,
        )

    def assert_bound(self, run_dir: Path) -> dict:
        approval = json.loads((run_dir / "plan_approval.json").read_text())
        self.assertEqual(approval["schema_version"], 2)
        self.assertEqual(
            approval["execution_sha256"], sha256_file(run_dir / "execution_selection.json")
        )
        return approval

    def test_profile_aware_approve_writes_schema_v2_bound_to_selection(self) -> None:
        run_dir = awaiting_run(self.runs, "modern")
        self.assertEqual(self.approve("modern")["decision"], "APPROVE")
        approval = self.assert_bound(run_dir)
        selection = read_execution_selection(run_dir)
        self.assertEqual(selection.schema_version, 2)
        self.assertEqual(selection.implementer.profile_id, "impl-b")
        state = RunStateStore(run_dir / "state.json").load()
        self.assertEqual(state["status"], RunStatus.AWAITING_PLAN_APPROVAL.value)
        self.assertEqual(state["execution"]["implementer"]["profile_id"], "impl-b")
        self.assertEqual(state["plan_identity"]["execution_sha256"], approval["execution_sha256"])
        identity = compute_plan_identity(PLAN, "# contract\n")
        self.assertEqual(
            read_plan_approval(run_dir, expected_identity=identity).execution_sha256,
            approval["execution_sha256"],
        )

    def test_profile_aware_approve_requires_profiles_and_writes_nothing(self) -> None:
        run_dir = awaiting_run(self.runs, "missing")
        for kwargs in ({}, {"implementer_profile": "impl-b"}, {"config": self.config}):
            with self.subTest(kwargs=kwargs), self.assertRaises(WebAPIError) as raised:
                approve_run(self.runs, "missing", "APPROVE", **kwargs)
            self.assertEqual(raised.exception.status, 400)
        with self.assertRaises(WebAPIError) as raised:
            self.approve("missing", implementer="review-a")
        self.assertEqual(raised.exception.status, 400)
        self.assertFalse((run_dir / "plan_approval.json").exists())
        self.assertFalse((run_dir / "execution_selection.json").exists())

    def test_cli_approve_refuses_profile_aware_run_and_reject_still_works(self) -> None:
        run_dir = awaiting_run(self.runs, "cli-modern")
        before = (run_dir / "state.json").read_bytes()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main(["approve-plan", "--run", str(run_dir)]), 2)
        self.assertIn("web UI", stderr.getvalue())
        self.assertFalse((run_dir / "plan_approval.json").exists())
        self.assertFalse((run_dir / "execution_selection.json").exists())
        self.assertEqual((run_dir / "state.json").read_bytes(), before)
        self.assertEqual(main(["reject-plan", "--run", str(run_dir)]), 0)
        approval = json.loads((run_dir / "plan_approval.json").read_text())
        self.assertEqual((approval["decision"], approval["schema_version"]), ("REJECT", 1))

    def test_web_reject_needs_no_execution_selection(self) -> None:
        run_dir = awaiting_run(self.runs, "web-reject")
        approve_run(self.runs, "web-reject", "REJECT")
        self.assertEqual(json.loads((run_dir / "plan_approval.json").read_text())["decision"], "REJECT")
        self.assertFalse((run_dir / "execution_selection.json").exists())

    def test_historic_schema_v1_approval_remains_valid(self) -> None:
        identity = compute_plan_identity(PLAN, "# contract\n")
        cli_run = awaiting_run(self.runs, "historic-cli", profile_aware=False)
        self.assertEqual(main(["approve-plan", "--run", str(cli_run)]), 0)
        web_run = awaiting_run(self.runs, "historic-web", profile_aware=False)
        approve_run(self.runs, "historic-web", "APPROVE", config=self.config)
        for run_dir in (cli_run, web_run):
            with self.subTest(run=run_dir.name):
                payload = json.loads((run_dir / "plan_approval.json").read_text())
                self.assertEqual(payload["schema_version"], 1)
                self.assertNotIn("execution_sha256", payload)
                approval = read_plan_approval(run_dir, expected_identity=identity)
                self.assertIs(approval.decision, ApprovalDecision.APPROVE)
                self.assertFalse((run_dir / "execution_selection.json").exists())

    def test_second_decision_is_a_conflict_and_changes_nothing(self) -> None:
        run_dir = awaiting_run(self.runs, "second")
        self.approve("second", implementer="impl-a")
        files = {
            name: (run_dir / name).read_bytes()
            for name in ("plan_approval.json", "execution_selection.json")
        }
        for implementer in ("impl-b", "impl-a"):
            with self.subTest(implementer=implementer), self.assertRaises(WebAPIError) as raised:
                self.approve("second", implementer=implementer)
            self.assertEqual(raised.exception.status, 409)
        with self.assertRaises(WebAPIError) as raised:
            approve_run(self.runs, "second", "REJECT")
        self.assertEqual(raised.exception.status, 409)
        for name, content in files.items():
            self.assertEqual((run_dir / name).read_bytes(), content)
        self.assert_bound(run_dir)

    def test_claimed_selection_without_decision_only_accepts_the_same_choice(self) -> None:
        # A request that claimed the selection but did not publish a decision
        # (crash, or a concurrent loser) never lets another choice replace it.
        run_dir = awaiting_run(self.runs, "claimed")
        ensure_execution_selection(run_dir, selection_for(self.config, "impl-a", "review-b"))
        with self.assertRaises(WebAPIError) as raised:
            self.approve("claimed", implementer="impl-b")
        self.assertEqual(raised.exception.status, 409)
        self.assertFalse((run_dir / "plan_approval.json").exists())
        self.approve("claimed", implementer="impl-a")
        self.assertEqual(read_execution_selection(run_dir).implementer.profile_id, "impl-a")
        self.assert_bound(run_dir)

    def test_selection_mutation_after_approval_invalidates_it(self) -> None:
        run_dir = awaiting_run(self.runs, "mutated")
        self.approve("mutated")
        path = run_dir / "execution_selection.json"
        content = path.read_text().replace('"impl-b"', '"impl-a"')
        path.unlink()
        path.write_text(content)
        with self.assertRaisesRegex(Exception, "does not match approval"):
            read_plan_approval(run_dir, expected_identity=compute_plan_identity(PLAN, "# contract\n"))

    def test_concurrent_different_approvals_leave_one_immutable_selection(self) -> None:
        # Repeated so that both interleavings occur; nothing depends on order.
        for attempt in range(12):
            run_id = f"race-{attempt}"
            run_dir = awaiting_run(self.runs, run_id)
            barrier = threading.Barrier(2)
            outcomes: dict[str, Any] = {}

            def decide(implementer: str) -> None:
                barrier.wait()
                try:
                    outcomes[implementer] = self.approve(run_id, implementer, "review-a")
                except WebAPIError as exc:
                    outcomes[implementer] = exc

            threads = [threading.Thread(target=decide, args=(i,)) for i in ("impl-a", "impl-b")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            with self.subTest(attempt=attempt):
                winners = [i for i, o in outcomes.items() if isinstance(o, dict)]
                losers = [o for o in outcomes.values() if isinstance(o, WebAPIError)]
                self.assertEqual(len(winners), 1)
                self.assertEqual([loser.status for loser in losers], [409])
                self.assert_bound(run_dir)
                selection = read_execution_selection(run_dir)
                self.assertEqual(selection.implementer.profile_id, winners[0])
                self.assertEqual(selection.reviewer.profile_id, "review-a")
                loser_id = "impl-b" if winners[0] == "impl-a" else "impl-a"
                frozen = (run_dir / "execution_selection.json").read_bytes()
                with self.assertRaises(WebAPIError) as raised:
                    self.approve(run_id, loser_id, "review-a")
                self.assertEqual(raised.exception.status, 409)
                with self.assertRaises(ExecutionSelectionConflict):
                    ensure_execution_selection(run_dir, selection_for(self.config, loser_id))
                self.assertEqual((run_dir / "execution_selection.json").read_bytes(), frozen)
                self.assert_bound(run_dir)


# ------------------------------------------------------------------- runtime


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class FakeChatClient:
    """Stands in for OpenAIChatTextClient; records the endpoint it was built from."""

    instances: list["FakeChatClient"] = []

    def __init__(self, config: LLMEndpointConfig) -> None:
        self.config = config
        self.prompts: list[str] = []
        FakeChatClient.instances.append(self)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return PLAN if self.config.endpoint_path == "/planner" else PASS_REVIEW


class FakeCodex:
    """Stands in for CodexAgent; records the AgentConfig it was built from."""

    instances: list["FakeCodex"] = []

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.calls = 0
        FakeCodex.instances.append(self)

    def run(self, contract: str, worktree: Path, run_dir: Path, *, base_sha: str, env: dict) -> AgentResult:
        self.calls += 1
        (Path(worktree) / "feature.txt").write_text("implemented\n", encoding="utf-8")
        return AgentResult(0, False, "done\n", {}, "")


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "MetaHarness Test")
        git(self.repo, "config", "user.email", "test@example.invalid")
        (self.repo / "README.md").write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "base")
        self.config = make_config(self.root)
        FakeChatClient.instances = []
        FakeCodex.instances = []

    def run_with_gate(
        self, config: HarnessConfig, run_id: str, at_gate: Callable[[Path], None]
    ):
        real_wait = approval_module.wait_for_plan_approval
        observed: list[str] = []

        def hooked(run_dir: Path, *, identity: PlanIdentity, poll_interval_seconds: float):
            observed.append(RunStateStore(Path(run_dir) / "state.json").load()["status"])
            self.assertFalse((config.worktrees_root / run_id).exists())
            at_gate(Path(run_dir))
            return real_wait(run_dir, identity=identity, poll_interval_seconds=poll_interval_seconds)

        with mock.patch("metaharness.orchestrator.OpenAIChatTextClient", FakeChatClient), \
                mock.patch("metaharness.orchestrator.CodexAgent", FakeCodex), \
                mock.patch("metaharness.orchestrator.wait_for_plan_approval", side_effect=hooked):
            result = Orchestrator(config).run_text("Implement the feature.\n", run_id=run_id)
        self.assertEqual(observed, [RunStatus.AWAITING_PLAN_APPROVAL.value])
        return result

    def web_approve(self, config: HarnessConfig, run_id: str, implementer: str, reviewer: str) -> None:
        approve_run(
            config.runs_root,
            run_id,
            "APPROVE",
            config=config,
            implementer_profile=implementer,
            reviewer_profile=reviewer,
        )

    def reviewer_clients(self) -> list[FakeChatClient]:
        return [c for c in FakeChatClient.instances if c.config.endpoint_path != "/planner"]

    def assert_nothing_executed(self, run_id: str) -> None:
        self.assertFalse((self.config.worktrees_root / run_id).exists())
        self.assertEqual(FakeCodex.instances, [])
        self.assertEqual(self.reviewer_clients(), [])
        self.assertEqual(git(self.repo, "branch", "--list", "harness/*"), "")
        self.assertEqual(git(self.repo, "rev-list", "--all", "--count"), "1")

    def test_approved_profiles_reach_codex_and_reviewer(self) -> None:
        result = self.run_with_gate(
            self.config,
            "selected",
            lambda _run_dir: self.web_approve(self.config, "selected", "impl-b", "review-b"),
        )
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        self.assertTrue((self.config.worktrees_root / "selected").exists())
        self.assertEqual(len(FakeCodex.instances), 1)
        codex = FakeCodex.instances[0]
        self.assertEqual(codex.calls, 1)
        self.assertEqual(
            (codex.config.model, codex.config.effort, codex.config.sandbox, codex.config.timeout_seconds),
            ("codex-impl-b", "high", "workspace-write", 30),
        )
        self.assertEqual(codex.config.env_allowlist, ENV_ALLOWLIST)
        reviewers = self.reviewer_clients()
        self.assertEqual([c.config.endpoint_path for c in reviewers], ["/review-b"])
        self.assertEqual(reviewers[0].config.model, "model-review-b")
        self.assertEqual(len(reviewers[0].prompts), 1)
        state = result.state
        self.assertEqual(state["execution"]["implementer"]["profile_id"], "impl-b")
        self.assertEqual(state["execution"]["reviewer"]["profile_id"], "review-b")
        run_dir = self.config.runs_root / "selected"
        self.assertEqual(
            state["plan_identity"]["execution_sha256"],
            sha256_file(run_dir / "execution_selection.json"),
        )
        self.assertEqual(git(self.config.worktrees_root / "selected", "rev-list", "--count", "HEAD"), "2")
        body = git(self.config.worktrees_root / "selected", "log", "-1", "--format=%B")
        self.assertIn("Implementer profile: impl-b", body)
        self.assertIn("Reviewer profile: review-b", body)

    def test_fingerprint_mismatch_stops_before_worktree_agent_reviewer_and_commit(self) -> None:
        # The UI approved against one configuration; the runtime now holds a
        # different execution profile under the same id.
        for field, value in (("model", "codex-impl-b-changed"), ("effort", "low")):
            with self.subTest(field=field):
                FakeChatClient.instances = []
                FakeCodex.instances = []
                run_id = f"mismatch-{field}"
                runtime = replace_profile(self.config, "impl-b", **{field: value})
                result = self.run_with_gate(
                    runtime,
                    run_id,
                    lambda _run_dir: self.web_approve(self.config, run_id, "impl-b", "review-b"),
                )
                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertEqual(result.state["failure"]["reason"], "EXECUTION_SELECTION_INVALID")
                self.assert_nothing_executed(run_id)

    def test_env_allowlist_change_stops_before_worktree(self) -> None:
        runtime = dataclasses.replace(
            self.config, agent=AgentConfig(env_allowlist=ENV_ALLOWLIST + ("OPENAI_API_KEY",))
        )
        result = self.run_with_gate(
            runtime,
            "allowlist",
            lambda _run_dir: self.web_approve(self.config, "allowlist", "impl-b", "review-b"),
        )
        self.assertEqual(result.state["failure"]["reason"], "EXECUTION_SELECTION_INVALID")
        self.assert_nothing_executed("allowlist")

    def test_profile_aware_schema_v1_approval_fails_before_worktree(self) -> None:
        def v1_approval(run_dir: Path) -> None:
            write_plan_approval(
                run_dir,
                decision=ApprovalDecision.APPROVE,
                identity=compute_plan_identity_from_run(run_dir),
                source="test",
            )
            self.assertEqual(json.loads((run_dir / "plan_approval.json").read_text())["schema_version"], 1)

        result = self.run_with_gate(self.config, "v1", v1_approval)
        self.assertEqual(result.state["failure"]["reason"], "PLAN_APPROVAL_INVALID")
        self.assertFalse((self.config.runs_root / "v1" / "execution_selection.json").exists())
        self.assert_nothing_executed("v1")

    def test_v1_approval_with_defaults_selection_present_still_fails(self) -> None:
        # Even a valid default selection cannot be activated by a v1 approval.
        def v1_with_selection(run_dir: Path) -> None:
            identity = compute_plan_identity_from_run(run_dir)
            ensure_execution_selection(run_dir, selection_for(self.config))
            write_plan_approval(
                run_dir, decision=ApprovalDecision.APPROVE, identity=identity, source="test"
            )

        result = self.run_with_gate(self.config, "v1-selection", v1_with_selection)
        self.assertEqual(result.state["failure"]["reason"], "PLAN_APPROVAL_INVALID")
        self.assert_nothing_executed("v1-selection")

    def test_selection_mutated_after_approval_fails_before_worktree(self) -> None:
        def approve_then_mutate(run_dir: Path) -> None:
            self.web_approve(self.config, "mutated", "impl-b", "review-b")
            path = run_dir / "execution_selection.json"
            content = path.read_text().replace('"impl-b"', '"impl-a"')
            path.unlink()
            path.write_text(content)

        result = self.run_with_gate(self.config, "mutated", approve_then_mutate)
        self.assertEqual(result.state["failure"]["reason"], "PLAN_APPROVAL_INVALID")
        self.assert_nothing_executed("mutated")


# --------------------------------------------------------------- run manager


class RunManagerHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = make_config(self.root, max_active_runs=2)
        self.config.runs_root.mkdir()
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        root = self.root
        release = self.release

        class Blocking:
            def __init__(self, _config: HarnessConfig) -> None:
                pass

            def run_text(self, _spec: str, *, run_id: str, on_created, **_kwargs: Any) -> None:
                on_created(root / "runs" / run_id)
                release.wait(timeout=5)

        self.manager = RunManager(self.config, max_active_runs=2, orchestrator_factory=Blocking)

    def wait_idle(self, manager: RunManager) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and manager._active_run_ids:
            time.sleep(0.01)
        self.assertEqual(manager._active_run_ids, set())

    def test_same_active_run_id_is_rejected_before_capacity(self) -> None:
        self.assertEqual(self.manager.start_run("spec", run_id="one"), "one")
        with self.assertRaises(RunCollisionError):
            self.manager.start_run("spec", run_id="one")
        with self.assertRaises(WebAPIError) as raised:
            create_run(self.manager, spec="spec", run_id="one")
        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(raised.exception.message, "run already exists or is active")
        self.assertEqual(self.manager.start_run("spec", run_id="two"), "two")
        # At capacity, a duplicate id is still reported as a collision.
        with self.assertRaises(RunCollisionError):
            self.manager.start_run("spec", run_id="two")
        with self.assertRaises(RunCapacityError):
            self.manager.start_run("spec", run_id="three")
        self.assertEqual(self.manager._active_run_ids, {"one", "two"})

    def test_fast_factory_failure_returns_immediately_and_restores_capacity(self) -> None:
        def failing_factory(_config: HarnessConfig):
            raise RuntimeError("factory exploded")

        class FailsBeforeCreation:
            def __init__(self, _config: HarnessConfig) -> None:
                pass

            def run_text(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("worker exploded")

        for factory in (failing_factory, FailsBeforeCreation):
            with self.subTest(factory=getattr(factory, "__name__", "factory")):
                manager = RunManager(self.config, max_active_runs=2, orchestrator_factory=factory)
                started = time.monotonic()
                with self.assertRaisesRegex(RunManagerError, "failed before durable creation"):
                    manager.start_run("spec", run_id="fails")
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertEqual(manager._active_run_ids, set())

    def test_capacity_is_restored_after_runs_finish(self) -> None:
        self.manager.start_run("spec", run_id="one")
        self.manager.start_run("spec", run_id="two")
        with self.assertRaises(RunCapacityError):
            self.manager.start_run("spec", run_id="three")
        self.release.set()
        self.wait_idle(self.manager)
        self.release.clear()
        self.assertEqual(self.manager.start_run("spec", run_id="three"), "three")
        self.assertEqual(self.manager.start_run("spec", run_id="four"), "four")
        self.release.set()
        self.wait_idle(self.manager)


if __name__ == "__main__":
    unittest.main()
