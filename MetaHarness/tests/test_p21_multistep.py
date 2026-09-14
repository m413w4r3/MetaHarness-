"""P21 staged execution end to end: fake planner, fake Codex, fake reviewer.

Every test uses real temporary Git repositories, the real approval API and
the real orchestrator.  No provider or network endpoint is contacted.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness import orchestrator as orchestrator_module  # noqa: E402
from metaharness.agent.base import AgentResult  # noqa: E402
from metaharness.agent.codex import CodexAgent, build_implementer_step_prompt  # noqa: E402
from metaharness.config import load_config  # noqa: E402
from metaharness.execution_selection import (  # noqa: E402
    ExecutionSelectionError,
    ensure_execution_selection_v3,
    resolve_execution_selection_v3,
)
from metaharness.llm.chat import TextLLMResult  # noqa: E402
from metaharness.models import AgentConfig, RunStatus  # noqa: E402
from metaharness.orchestrator import Orchestrator  # noqa: E402
from metaharness.usage import USAGE_FIELDS  # noqa: E402
from metaharness.web.api import approve_run, get_run  # noqa: E402
from metaharness.web.pages import render_run  # noqa: E402


SPEC_MARKER = "SPEC-MARKER-7f3a"
CONTEXT_MARKER = "GLOBAL-CONTEXT-MARKER-91be"
PLAN_OBJECTIVE = "Implement the staged feature for the whole run."
SPEC = f"Please implement the staged feature. {SPEC_MARKER}\n"

PASS_REVIEW = """VERDICT: PASS
ROUTE: NONE
SUMMARY: The implementation is acceptable.
FINDINGS: NONE
REQUIRED FIXES: NONE
MISSING TESTS: NONE
RESIDUAL RISKS: NONE
"""

PLANNER_USAGE = {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500, "cached_input_tokens": 200}
REVIEWER_USAGE = {"prompt_tokens": 5000, "completion_tokens": 800, "reasoning_output_tokens": 120}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, shell=False
    ).stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def step_block(
    number: int,
    *,
    profile: str = "impl-a",
    read: tuple[str, ...] = ("src/a.py",),
    write_set: tuple[str, ...] = ("src/a.py",),
    create: tuple[str, ...] = (),
    delete: tuple[str, ...] = (),
    legacy: bool = False,
) -> str:
    step_id = f"S{number:02d}"
    dependency = "NONE" if number == 1 else f"S{number - 1:02d}"
    lines = [
        f"BEGIN STEP {step_id}",
        f"TITLE: Step {number} unique-title-{number}",
        f"IMPLEMENTER_PROFILE: {profile}",
        f"DEPENDS_ON: {dependency}",
        "",
        "OBJECTIVE",
        f"Objective of step {number} only.",
        "",
        "READ_SET",
        *[f"- {path} :: anchor-{number}" for path in read],
        "",
        "WRITE_SET",
        *([f"- {path}" for path in write_set] or ["NONE"]),
        "",
    ]
    if not legacy:
        lines += [
            "CREATE_SET",
            *([f"- {path}" for path in create] or ["NONE"]),
            "",
            "DELETE_SET",
            *([f"- {path}" for path in delete] or ["NONE"]),
            "",
        ]
    lines += [
        "INSTRUCTIONS",
        f"1. Perform operation {number} exactly.",
        "",
        "VERIFY",
        f"- python -m unittest tests.test_step_{number}",
        "",
        "FORBIDDEN",
        "- Do not touch any other file.",
        "",
        f"END STEP {step_id}",
    ]
    return "\n".join(lines)


def plan_text(*steps: str) -> str:
    mode = "SINGLE" if len(steps) == 1 else "STAGED"
    return "\n".join(
        [
            "META PLAN v2",
            "",
            "STATUS: READY",
            "TITLE: Staged feature",
            "",
            "OBJECTIVE",
            PLAN_OBJECTIVE,
            "",
            "CONSTRAINTS",
            "NONE",
            "",
            f"EXECUTION_MODE: {mode}",
            f"STEP_COUNT: {len(steps)}",
            "REVIEWER_PROFILE: reviewer",
            "",
            "\n\n".join(steps),
            "",
            "ACCEPTANCE",
            "The feature is observable.",
            "",
            "TESTS",
            "Run the configured check.",
            "",
            "RISKS",
            "NONE",
            "",
            "BLOCKERS",
            "NONE",
            "",
            "END META PLAN",
            "",
        ]
    )


class FakeClient:
    """Planner or reviewer double returning one fixed answer with usage."""

    def __init__(self, text: str, usage: dict[str, int]):
        self.text = text
        self.usage = usage
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> TextLLMResult:
        self.prompts.append(prompt)
        return TextLLMResult(text=self.text, model="fake", usage=dict(self.usage), raw_response={})


def step_usage(number: int) -> dict[str, int]:
    return {
        "input_tokens": 1000 * number,
        "cached_input_tokens": 100 * number,
        "output_tokens": 50 * number,
        "reasoning_output_tokens": 5 * number,
        "total_tokens": 1050 * number,
    }


class FakeAgent:
    """In-process Codex double; each ``run_step`` is one fresh invocation."""

    def __init__(
        self,
        actions: dict[str, Callable[[Path], Any]] | None = None,
        *,
        usage: dict[str, dict[str, int]] | None = None,
    ):
        self.actions = actions or {}
        self.usage = usage or {}
        self.calls: list[dict[str, Any]] = []

    def run_step(self, contract: str, worktree: Any, artifacts_dir: Any, *, base_sha: str | None = None,
                 env: dict[str, str] | None = None) -> AgentResult:
        step_id = re.search(r"^STEP\n(S0[1-6]) / ", contract, re.M).group(1)
        root = Path(worktree)
        seen = {
            name: (root / name).read_text(encoding="utf-8")
            for name in ("src/a.py", "src/b.py", "src/c.py")
            if (root / name).exists()
        }
        self.calls.append({
            "step": step_id, "contract": contract,
            "prompt": build_implementer_step_prompt(contract),
            "env": dict(env or {}), "seen": seen, "artifacts_dir": Path(artifacts_dir),
        })
        outcome = self.actions[step_id](root) if step_id in self.actions else None
        usage = self.usage.get(step_id, step_usage(int(step_id[1:])))
        directory = Path(artifacts_dir)
        directory.mkdir(parents=True, exist_ok=True)
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": f"working on {step_id}"}},
            {"type": "item.started", "item": {"type": "command_execution", "command": "bash -lc 'cat src/a.py --secret-arg'"}},
            {"type": "turn.completed", "usage": usage},
        ]
        (directory / "agent.events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        (directory / "agent.final.md").write_text(f"{step_id} done\n", encoding="utf-8")
        timed_out = outcome == "timeout"
        return AgentResult(
            exit_code=124 if timed_out else 0, timed_out=timed_out,
            final_message=f"{step_id} done\n", usage=dict(usage), stderr_tail="",
        )


class RecordingOrchestrator(Orchestrator):
    """Records the implementer profile resolved for every fresh step agent."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_profiles: list[str] = []

    def _agent_for_profile(self, profile_id: str):  # type: ignore[override]
        self.agent_profiles.append(profile_id)
        return super()._agent_for_profile(profile_id)


class MultiStepHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "MetaHarness P21")
        git(self.repo, "config", "user.email", "p21@example.invalid")
        write(self.repo / "README.md", f"Project readme {CONTEXT_MARKER}\n")
        write(self.repo / "src/a.py", "A = 1\n")
        write(self.repo / "src/b.py", "B = 1\n")
        write(self.repo / "src/old.py", "OLD = 1\n")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "base")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.runs = self.root / "runs"
        self.counter = self.root / "check-count.txt"
        self.check = self.root / "check.py"
        self.check.write_text(
            "import sys\nopen(sys.argv[1], 'a').write('ran\\n')\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, *, require_approval: bool):
        path = self.root / "config.toml"
        path.write_text(
            textwrap.dedent(
                f"""
                repo = {str(self.repo)!r}
                base_ref = "HEAD"
                runs_root = {str(self.runs)!r}
                worktrees_root = {str(self.root / 'worktrees')!r}
                require_clean_base = true

                [planning]
                protocol = "v2"

                [codex_runtime]
                home = {str(self.root / 'codex-home')!r}

                [approval]
                require_plan_approval = {'true' if require_approval else 'false'}
                poll_interval_seconds = 0.01

                [ui]
                enable_profile_recommendation = false
                default_planner_profile = "planner"
                default_implementer_profile = "impl-a"
                default_reviewer_profile = "reviewer"

                [context]
                always_files = ["README.md"]

                [model_profiles.planner]
                display_name = "Planner"
                roles = ["planner"]
                driver = "openai-chat"
                model = "fake-planner"
                selection_mode = "request"
                base_url = "http://127.0.0.1:9"
                endpoint_path = "/v1/chat/completions"
                retries = 0

                [model_profiles.reviewer]
                display_name = "Reviewer"
                roles = ["reviewer"]
                driver = "openai-chat"
                model = "fake-reviewer"
                selection_mode = "request"
                base_url = "http://127.0.0.1:9"
                endpoint_path = "/v1/chat/completions"
                retries = 0

                [model_profiles."impl-a"]
                display_name = "Impl A"
                roles = ["implementer"]
                driver = "codex"
                model = "luna-a"
                effort = "high"
                sandbox = "workspace-write"
                selection_mode = "cli"
                timeout_seconds = 30

                [model_profiles."impl-b"]
                display_name = "Impl B"
                roles = ["implementer"]
                driver = "codex"
                model = "luna-b"
                effort = "low"
                sandbox = "workspace-write"
                selection_mode = "cli"
                timeout_seconds = 30

                [[checks]]
                name = "count"
                argv = [{sys.executable!r}, {str(self.check)!r}, {str(self.counter)!r}]
                timeout_seconds = 30
                """
            ),
            encoding="utf-8",
        )
        return load_config(path)

    def run_v2(
        self,
        plan: str,
        agent: Any,
        *,
        approve: Callable[[Any, str], Any] | None = None,
        tamper: Callable[[Path], None] | None = None,
        review: str = PASS_REVIEW,
        run_id: str = "run-1",
    ):
        config = self.config(require_approval=approve is not None)
        self.config_value = config
        planner = FakeClient(plan, PLANNER_USAGE)
        reviewer = FakeClient(review, REVIEWER_USAGE)
        orchestrator = RecordingOrchestrator(
            config, planner_client=planner, reviewer_client=reviewer, agent=agent
        )
        if approve is None:
            return orchestrator.run_text(SPEC, run_id=run_id), orchestrator, planner, reviewer
        run_dir = self.runs / run_id
        real_wait = orchestrator_module.wait_for_plan_approval

        def wait_then_tamper(*args: Any, **kwargs: Any):
            approval = real_wait(*args, **kwargs)
            tamper(run_dir)
            return approval

        holder: dict[str, Any] = {}
        patcher = (
            mock.patch.object(orchestrator_module, "wait_for_plan_approval", side_effect=wait_then_tamper)
            if tamper is not None
            else nullcontext()
        )
        with patcher:
            thread = threading.Thread(
                target=lambda: holder.setdefault("result", orchestrator.run_text(SPEC, run_id=run_id))
            )
            thread.start()
            deadline = time.monotonic() + 15
            state_path = run_dir / "state.json"
            while time.monotonic() < deadline:
                if state_path.exists() and json.loads(state_path.read_text())["status"] == "awaiting_plan_approval":
                    break
                time.sleep(0.01)
            self.assertEqual(json.loads(state_path.read_text())["status"], "awaiting_plan_approval")
            self.assertFalse((self.root / "worktrees" / run_id).exists())
            self.awaiting_run = get_run(self.runs, run_id)
            self.awaiting_page = render_run(self.awaiting_run, "token", config=config)
            approve(config, run_id)
            thread.join(timeout=30)
            self.assertFalse(thread.is_alive())
        return holder["result"], orchestrator, planner, reviewer

    def checks_run(self) -> int:
        if not self.counter.exists():
            return 0
        return len(self.counter.read_text().splitlines())

    def state(self, run_id: str = "run-1") -> dict[str, Any]:
        return json.loads((self.runs / run_id / "state.json").read_text())

    def worktree(self, run_id: str = "run-1") -> Path:
        return self.root / "worktrees" / run_id

    def contract(self, step_id: str, run_id: str = "run-1") -> str:
        return (self.runs / run_id / "steps" / step_id / "contract.md").read_text(encoding="utf-8")

    def assert_stopped_before_final_gates(self, reviewer: FakeClient) -> None:
        self.assertEqual(self.checks_run(), 0)
        self.assertEqual(reviewer.prompts, [])
        self.assertFalse((self.runs / "run-1" / "reviewer.raw.md").exists())


def approve_with(step_profiles: dict[str, str]):
    def approve(config: Any, run_id: str) -> Any:
        return approve_run(
            config.runs_root, run_id, "APPROVE", config=config,
            reviewer_profile="reviewer", step_profiles=step_profiles,
        )

    return approve


def three_steps() -> str:
    return plan_text(
        step_block(1),
        step_block(2, read=("src/a.py", "src/b.py"), write_set=("src/b.py",)),
        step_block(3, read=("src/a.py",), write_set=(), create=("src/c.py",)),
    )


def three_step_actions() -> dict[str, Callable[[Path], Any]]:
    return {
        "S01": lambda wt: write(wt / "src/a.py", "A = 2\n"),
        "S02": lambda wt: write(wt / "src/b.py", "B = 2\n"),
        "S03": lambda wt: write(wt / "src/c.py", "C = 3\n"),
    }


class SingleAndStagedTests(MultiStepHarness):
    def test_single_runs_one_planner_one_agent_one_check_one_review_one_commit(self) -> None:
        agent = FakeAgent({"S01": lambda wt: write(wt / "src/a.py", "A = 2\n")})
        result, orchestrator, planner, reviewer = self.run_v2(plan_text(step_block(1)), agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        self.assertEqual(len(planner.prompts), 1)
        self.assertEqual([call["step"] for call in agent.calls], ["S01"])
        self.assertEqual(orchestrator.agent_profiles, ["impl-a"])
        self.assertEqual(self.checks_run(), 1)
        self.assertEqual(len(reviewer.prompts), 1)
        worktree = self.worktree()
        self.assertEqual(git(worktree, "rev-list", "--count", "HEAD"), "2")
        self.assertEqual(git(worktree, "rev-parse", "HEAD^"), self.base_sha)
        self.assertEqual(git(worktree, "rev-parse", "HEAD^{tree}"), result.state["staged_tree_sha"])
        self.assertEqual(git(worktree, "diff", "--name-only", self.base_sha, "HEAD"), "src/a.py")
        self.assertIn("S01 implementer profile: impl-a", git(worktree, "log", "-1", "--format=%B"))
        self.assertEqual(result.state["steps"][0]["status"], "completed")
        self.assertIsNone(result.state["current_step"])

    def test_staged_three_steps_run_in_order_with_one_final_gate(self) -> None:
        agent = FakeAgent(three_step_actions())
        result, orchestrator, planner, reviewer = self.run_v2(three_steps(), agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        self.assertEqual([call["step"] for call in agent.calls], ["S01", "S02", "S03"])
        # One fresh agent resolution and one run_step invocation per step.
        self.assertEqual(orchestrator.agent_profiles, ["impl-a", "impl-a", "impl-a"])
        self.assertEqual(len({id(call["artifacts_dir"]) for call in agent.calls}), 3)
        self.assertEqual([call["artifacts_dir"].name for call in agent.calls], ["S01", "S02", "S03"])
        # Earlier step changes are visible to later steps.
        self.assertEqual(agent.calls[0]["seen"]["src/a.py"], "A = 1\n")
        self.assertEqual(agent.calls[1]["seen"]["src/a.py"], "A = 2\n")
        self.assertEqual(agent.calls[2]["seen"]["src/b.py"], "B = 2\n")
        self.assertEqual(self.checks_run(), 1)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(len(planner.prompts), 1)
        worktree = self.worktree()
        self.assertEqual(git(worktree, "rev-list", "--count", "HEAD"), "2")
        self.assertEqual(
            git(worktree, "diff", "--name-only", self.base_sha, "HEAD").splitlines(),
            ["src/a.py", "src/b.py", "src/c.py"],
        )
        self.assertEqual([item["status"] for item in result.state["steps"]], ["completed"] * 3)
        for step_id in ("S01", "S02", "S03"):
            step = json.loads((self.runs / "run-1" / "steps" / step_id / "step.json").read_text())
            self.assertEqual(step["status"], "COMPLETED")
            self.assertEqual(agent.calls[int(step_id[1:]) - 1]["contract"], self.contract(step_id))


class ApprovalTests(MultiStepHarness):
    def test_exact_contracts_are_shown_and_s02_override_is_executed(self) -> None:
        agent = FakeAgent(three_step_actions())
        # Non-canonical insertion order: the durable selection is S01..S03.
        approve = approve_with({"S03": "impl-a", "S02": "impl-b", "S01": "impl-a"})
        result, orchestrator, _planner, _reviewer = self.run_v2(three_steps(), agent, approve=approve)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        run_dir = self.runs / "run-1"
        page = self.awaiting_page
        self.assertIn("Execution mode: <strong>STAGED</strong>", page)
        self.assertIn("Steps: 3", page)
        for step_id in ("S01", "S02", "S03"):
            contract = self.contract(step_id)
            self.assertIn(html.escape(contract, quote=True), page)
            artifact = next(item for item in self.awaiting_run["step_artifacts"] if item["id"] == step_id)
            self.assertEqual(artifact["contract"], contract)
            self.assertTrue(artifact["contract_matches_bundle"])
            self.assertFalse((run_dir / "steps" / f"{step_id}.contract.md").exists())
        self.assertIn("Exact implementation contract", page)
        self.assertIn("Recommended implementer", page)
        # What the human read is what Luna received.
        self.assertEqual([call["contract"] for call in agent.calls],
                         [self.contract(step_id) for step_id in ("S01", "S02", "S03")])
        self.assertEqual(orchestrator.agent_profiles, ["impl-a", "impl-b", "impl-a"])
        selection = json.loads((run_dir / "execution_selection.json").read_text())
        self.assertEqual([item["step_id"] for item in selection["steps"]], ["S01", "S02", "S03"])
        self.assertEqual(selection["steps"][1]["implementer"]["profile_id"], "impl-b")
        step_two = json.loads((run_dir / "steps" / "S02" / "step.json").read_text())
        self.assertEqual(step_two["profile_id"], "impl-b")
        self.assertIn("S02 implementer profile: impl-b", git(self.worktree(), "log", "-1", "--format=%B"))
        page_after = render_run(get_run(self.runs, "run-1"), "token", config=self.config_value)
        self.assertIn("approved <span class=\"mono\">impl-b / luna-b / low</span>", page_after)

    def _assert_invalid_before_worktree(self, tamper: Callable[[Path], None]) -> None:
        agent = FakeAgent(three_step_actions())
        approve = approve_with({"S01": "impl-a", "S02": "impl-b", "S03": "impl-a"})
        result, _orchestrator, _planner, reviewer = self.run_v2(
            three_steps(), agent, approve=approve, tamper=tamper
        )
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "PLAN_APPROVAL_INVALID")
        self.assertFalse(self.worktree().exists())
        self.assertEqual(agent.calls, [])
        self.assertEqual(reviewer.prompts, [])
        self.assertEqual(self.checks_run(), 0)

    def test_bundle_tamper_after_approval_fails_before_worktree(self) -> None:
        def tamper(run_dir: Path) -> None:
            path = run_dir / "implementation_bundle.json"
            path.write_text(path.read_text() + " ", encoding="utf-8")

        self._assert_invalid_before_worktree(tamper)

    def test_contract_tamper_after_approval_fails_before_worktree(self) -> None:
        def tamper(run_dir: Path) -> None:
            path = run_dir / "steps" / "S02" / "contract.md"
            data = bytearray(path.read_bytes())
            index = data.index(b"Perform")
            data[index] = ord("p")
            path.write_bytes(bytes(data))

        self._assert_invalid_before_worktree(tamper)

    def test_execution_selection_tamper_after_approval_fails_before_worktree(self) -> None:
        def tamper(run_dir: Path) -> None:
            path = run_dir / "execution_selection.json"
            payload = json.loads(path.read_text())
            payload["steps"][1]["implementer"] = payload["steps"][0]["implementer"]
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        self._assert_invalid_before_worktree(tamper)


class StepGateTests(MultiStepHarness):
    def test_no_change_fails_with_agent_no_change(self) -> None:
        agent = FakeAgent({})
        result, _orchestrator, _planner, reviewer = self.run_v2(plan_text(step_block(1)), agent)
        self.assertEqual(result.state["failure"]["reason"], "AGENT_NO_CHANGE")
        self.assertEqual(result.state["failure"]["detail"], "step=S01")
        self.assert_stopped_before_final_gates(reviewer)

    def test_unexpected_modification_fails_with_write_set_violation(self) -> None:
        def action(wt: Path) -> None:
            write(wt / "src/a.py", "A = 2\n")
            write(wt / "src/b.py", "B = 99\n")

        result, _orchestrator, _planner, reviewer = self.run_v2(plan_text(step_block(1)), FakeAgent({"S01": action}))
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "STEP_WRITE_SET_VIOLATION")
        self.assertEqual(result.state["failure"]["detail"], "step=S01 unexpected=src/b.py")
        self.assert_stopped_before_final_gates(reviewer)
        self.assertEqual(result.state["steps"][0]["status"], "failed")

    def test_unexpected_new_file_is_a_violation(self) -> None:
        def action(wt: Path) -> None:
            write(wt / "src/a.py", "A = 2\n")
            write(wt / "notes/scratch.txt", "x\n")

        result, _orchestrator, _planner, reviewer = self.run_v2(plan_text(step_block(1)), FakeAgent({"S01": action}))
        self.assertEqual(result.state["failure"]["reason"], "STEP_WRITE_SET_VIOLATION")
        self.assertIn("unexpected=notes/scratch.txt", result.state["failure"]["detail"])
        self.assert_stopped_before_final_gates(reviewer)

    def test_missing_read_path_is_contract_drift_before_codex(self) -> None:
        plan = plan_text(step_block(1, read=("src/a.py", "src/missing.py")))
        agent = FakeAgent({"S01": lambda wt: write(wt / "src/a.py", "A = 2\n")})
        result, _orchestrator, _planner, reviewer = self.run_v2(plan, agent)
        self.assertEqual(result.state["failure"]["reason"], "STEP_CONTRACT_DRIFT")
        self.assertEqual(result.state["failure"]["detail"], "step=S01 read_missing=src/missing.py")
        self.assertEqual(agent.calls, [])
        self.assert_stopped_before_final_gates(reviewer)
        self.assertEqual(result.state["steps"][0]["status"], "failed")

    def test_existing_create_path_is_contract_drift(self) -> None:
        plan = plan_text(step_block(1, create=("src/b.py",)))
        agent = FakeAgent({"S01": lambda wt: write(wt / "src/a.py", "A = 2\n")})
        result, _orchestrator, _planner, reviewer = self.run_v2(plan, agent)
        self.assertEqual(result.state["failure"]["reason"], "STEP_CONTRACT_DRIFT")
        self.assertEqual(result.state["failure"]["detail"], "step=S01 create_exists=src/b.py")
        self.assertEqual(agent.calls, [])
        self.assert_stopped_before_final_gates(reviewer)

    def test_create_set_new_file_is_accepted(self) -> None:
        plan = plan_text(step_block(1, write_set=(), create=("src/new_module.py",)))
        agent = FakeAgent({"S01": lambda wt: write(wt / "src/new_module.py", "NEW = 1\n")})
        result, _orchestrator, _planner, _reviewer = self.run_v2(plan, agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        self.assertEqual(git(self.worktree(), "show", "HEAD:src/new_module.py"), "NEW = 1")

    def test_delete_set_deletion_is_accepted(self) -> None:
        plan = plan_text(step_block(1, read=("src/old.py",), write_set=(), delete=("src/old.py",)))
        agent = FakeAgent({"S01": lambda wt: (wt / "src/old.py").unlink()})
        result, _orchestrator, _planner, _reviewer = self.run_v2(plan, agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        self.assertEqual(git(self.worktree(), "ls-tree", "--name-only", "HEAD", "src/old.py"), "")

    def test_unused_create_and_delete_permissions_are_accepted(self) -> None:
        plan = plan_text(step_block(
            1, read=("src/a.py", "src/old.py"), create=("src/unused.py",), delete=("src/old.py",)
        ))
        agent = FakeAgent({"S01": lambda wt: write(wt / "src/a.py", "A = 2\n")})
        result, _orchestrator, _planner, _reviewer = self.run_v2(plan, agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))

    def test_agent_commit_is_rejected(self) -> None:
        def action(wt: Path) -> None:
            write(wt / "src/a.py", "A = 2\n")
            git(wt, "add", "--all")
            git(wt, "commit", "-qm", "agent-owned commit")

        result, _orchestrator, _planner, reviewer = self.run_v2(plan_text(step_block(1)), FakeAgent({"S01": action}))
        self.assertEqual(result.state["failure"]["reason"], "AGENT_COMMITTED")
        self.assertTrue(result.state["failure"]["detail"].startswith("step=S01"))
        self.assert_stopped_before_final_gates(reviewer)

    def test_timeout_stops_following_steps(self) -> None:
        actions = three_step_actions()
        actions["S02"] = lambda wt: "timeout"
        agent = FakeAgent(actions)
        result, _orchestrator, _planner, reviewer = self.run_v2(three_steps(), agent)
        self.assertEqual(result.state["failure"]["reason"], "AGENT_TIMEOUT")
        self.assertEqual(result.state["failure"]["detail"], "step=S02")
        self.assertEqual([call["step"] for call in agent.calls], ["S01", "S02"])
        self.assertEqual([item["status"] for item in result.state["steps"]], ["completed", "failed", "waiting"])
        self.assertIsNone(result.state["current_step"])
        self.assert_stopped_before_final_gates(reviewer)

    def test_legacy_v2_plan_without_create_delete_sections_still_runs(self) -> None:
        plan = plan_text(step_block(1, legacy=True))
        agent = FakeAgent({"S01": lambda wt: write(wt / "src/a.py", "A = 2\n")})
        result, _orchestrator, _planner, _reviewer = self.run_v2(plan, agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        self.assertIn("CREATE SET\nNONE", self.contract("S01"))


class ScopeTests(MultiStepHarness):
    def test_worker_prompt_contains_only_its_own_contract(self) -> None:
        agent = FakeAgent(three_step_actions())
        result, _orchestrator, planner, _reviewer = self.run_v2(three_steps(), agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        # Sanity: the planner did receive SPEC and global context.
        self.assertIn(SPEC_MARKER, planner.prompts[0])
        self.assertIn(CONTEXT_MARKER, planner.prompts[0])
        first = agent.calls[0]["prompt"]
        self.assertEqual(first, build_implementer_step_prompt(self.contract("S01")))
        self.assertIn(self.contract("S01"), first)
        self.assertNotIn(SPEC_MARKER, first)
        self.assertNotIn(CONTEXT_MARKER, first)
        self.assertNotIn(PLAN_OBJECTIVE, first)
        self.assertNotIn("unique-title-2", first)
        self.assertNotIn(self.contract("S02"), first)
        self.assertNotIn("META PLAN v2", first)

    def test_real_codex_processes_are_fresh_and_receive_the_exact_prompt(self) -> None:
        pids = self.root / "pids.txt"
        script = self.root / "fake-codex"
        script.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                r'''
                import json, os, pathlib, re, sys
                prompt = sys.stdin.read()
                step = re.search(r"^STEP\n(S0[1-6]) / ", prompt, re.M).group(1)
                worktree = pathlib.Path(sys.argv[sys.argv.index("-C") + 1])
                with open(__PIDS__, "a") as stream:
                    stream.write(f"{step} {os.getpid()} {os.environ.get('CODEX_HOME', '')}\n")
                target = {"S01": "src/a.py", "S02": "src/b.py"}[step]
                (worktree / target).write_text(step + " done\n")
                final = pathlib.Path(sys.argv[sys.argv.index("--output-last-message") + 1])
                final.write_text(step + " report\n")
                print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": step + " finished"}}), flush=True)
                print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 11, "cached_input_tokens": 3, "output_tokens": 7, "reasoning_output_tokens": 2}}), flush=True)
                '''
            ).replace("__PIDS__", repr(str(pids))),
            encoding="utf-8",
        )
        script.chmod(0o755)
        agent = CodexAgent(AgentConfig(timeout_seconds=30), executable=str(script))
        plan = plan_text(step_block(1), step_block(2, read=("src/a.py", "src/b.py"), write_set=("src/b.py",)))
        result, _orchestrator, _planner, _reviewer = self.run_v2(plan, agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        records = [line.split(" ") for line in pids.read_text().splitlines()]
        self.assertEqual([record[0] for record in records], ["S01", "S02"])
        self.assertNotEqual(records[0][1], records[1][1])
        self.assertEqual(records[0][2], str((self.root / "codex-home").resolve()))
        for step_id in ("S01", "S02"):
            prompt = (self.runs / "run-1" / "steps" / step_id / "agent.prompt.txt").read_text()
            self.assertEqual(prompt, build_implementer_step_prompt(self.contract(step_id)))
            self.assertNotIn(SPEC_MARKER, prompt)
        step = json.loads((self.runs / "run-1" / "steps" / "S01" / "step.json").read_text())
        self.assertEqual(step["usage"], {
            "input_tokens": 11, "cached_input_tokens": 3, "cache_write_input_tokens": 0,
            "output_tokens": 7, "reasoning_output_tokens": 2, "total_tokens": 18,
        })


class TokenUsageTests(MultiStepHarness):
    def test_every_phase_is_persisted_and_aggregated(self) -> None:
        heavy = {**step_usage(2), "input_tokens": 150_000, "total_tokens": 150_100}
        agent = FakeAgent(three_step_actions(), usage={"S02": heavy})
        result, _orchestrator, _planner, _reviewer = self.run_v2(three_steps(), agent)
        self.assertEqual(result.status, RunStatus.COMMITTED, result.state.get("failure"))
        run_dir = self.runs / "run-1"
        planner_usage = json.loads((run_dir / "planner.usage.json").read_text())
        self.assertEqual(list(planner_usage), list(USAGE_FIELDS))
        self.assertEqual(planner_usage, {
            "input_tokens": 1200, "cached_input_tokens": 200, "cache_write_input_tokens": 0,
            "output_tokens": 300, "reasoning_output_tokens": 0, "total_tokens": 1500,
        })
        reviewer_usage = json.loads((run_dir / "reviewer.usage.json").read_text())
        self.assertEqual(reviewer_usage, {
            "input_tokens": 5000, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
            "output_tokens": 800, "reasoning_output_tokens": 120, "total_tokens": 5800,
        })
        expected_steps = {"S01": step_usage(1), "S02": {**heavy, "cache_write_input_tokens": 0},
                          "S03": step_usage(3)}
        for step_id, expected in expected_steps.items():
            persisted = json.loads((run_dir / "steps" / step_id / "step.json").read_text())["usage"]
            self.assertEqual(persisted, {**{name: 0 for name in USAGE_FIELDS}, **expected})
        usage = get_run(self.runs, "run-1")["usage"]
        self.assertEqual(usage["planner"], planner_usage)
        self.assertEqual(usage["reviewer"], reviewer_usage)
        self.assertEqual([item["id"] for item in usage["implementer"]["steps"]], ["S01", "S02", "S03"])
        luna_input = 1000 + 150_000 + 3000
        self.assertEqual(usage["implementer"]["total"]["input_tokens"], luna_input)
        self.assertEqual(usage["implementer"]["total"]["output_tokens"], 50 + 100 + 150)
        self.assertEqual(usage["implementer"]["total"]["cached_input_tokens"], 100 + 200 + 300)
        self.assertEqual(usage["grand_total"]["input_tokens"], 1200 + luna_input + 5000)
        self.assertEqual(usage["grand_total"]["output_tokens"], 300 + 300 + 800)
        page = render_run(get_run(self.runs, "run-1"), None, config=self.config_value)
        self.assertIn("TOKEN USAGE", page)
        self.assertIn("<th>Planner</th><td>1200 input / 300 output</td>", page)
        self.assertIn(f"<th>Luna</th><td>{luna_input} input / 300 output</td>", page)
        self.assertIn("<th>Reviewer</th><td>5000 input / 800 output</td>", page)
        self.assertIn("High worker context usage", page)
        # The warning is advisory: the run still committed.
        self.assertEqual(result.state["status"], "committed")


class UITests(MultiStepHarness):
    def test_codex_auth_failure_opens_agent_diagnostics_and_shows_login(self) -> None:
        config = self.config(require_approval=False)
        page = render_run(
            {
                "run_id": "auth-failure",
                "state": {
                    "status": "failed",
                    "planning_protocol": "v2",
                    "failure": {
                        "reason": "CODEX_AUTH_FAILURE",
                        "detail": "step=S01 Codex authentication failed",
                    },
                    "steps": [{"id": "S01", "status": "failed", "title": "Auth"}],
                },
                "failure": {
                    "reason": "CODEX_AUTH_FAILURE",
                    "detail": "step=S01 Codex authentication failed",
                },
            },
            config=config,
        )
        self.assertIn("Codex authentication failed.", page)
        self.assertIn(
            f'CODEX_HOME="{config.codex_runtime.home}" codex login',
            page,
        )
        self.assertIn('class="card step failed" open', page)

    def test_step_live_events_are_exposed_without_tool_arguments(self) -> None:
        agent = FakeAgent(three_step_actions())
        self.run_v2(three_steps(), agent)
        artifacts = get_run(self.runs, "run-1")["step_artifacts"]
        self.assertEqual([item["id"] for item in artifacts], ["S01", "S02", "S03"])
        events = artifacts[0]["events"]
        self.assertIn("message: working on S01", events)
        self.assertIn("tool: command bash", events)
        self.assertFalse(any("--secret-arg" in event for event in events))
        page = render_run(get_run(self.runs, "run-1"), None, config=self.config_value)
        self.assertIn("Recent events", page)
        self.assertIn("message: working on S02", page)
        self.assertNotIn("--secret-arg", page)

    def test_failed_step_is_rendered_failed_and_later_steps_wait(self) -> None:
        actions = three_step_actions()
        actions["S02"] = lambda wt: (write(wt / "src/b.py", "B = 2\n"), write(wt / "src/a.py", "A = 9\n"))
        result, _orchestrator, _planner, _reviewer = self.run_v2(three_steps(), FakeAgent(actions))
        self.assertEqual(result.state["failure"]["reason"], "STEP_WRITE_SET_VIOLATION")
        self.assertEqual(result.state["failure"]["detail"], "step=S02 unexpected=src/a.py")
        state = self.state()
        self.assertEqual([item["status"] for item in state["steps"]], ["completed", "failed", "waiting"])
        self.assertIsNone(state["current_step"])
        page = render_run(get_run(self.runs, "run-1"), None, config=self.config_value)
        self.assertIn("S01 ✓", page)
        self.assertIn("S02 ✗", page)
        self.assertIn("S03 …", page)
        self.assertNotIn("running", json.dumps(state["steps"]))

    def test_v2_execution_card_has_no_generic_implementer_card(self) -> None:
        agent = FakeAgent(three_step_actions())
        approve = approve_with({"S01": "impl-a", "S02": "impl-b", "S03": "impl-a"})
        self.run_v2(three_steps(), agent, approve=approve)
        for page in (self.awaiting_page, render_run(get_run(self.runs, "run-1"), None, config=self.config_value)):
            self.assertNotIn("<h3>Implementer</h3>", page)
            self.assertIn("<h3>Planner</h3>", page)
            self.assertIn("<h3>Reviewer</h3>", page)
            self.assertIn("Step implementers", page)
            self.assertIn("recommended <span class=\"mono\">impl-a / luna-a / high</span>", page)
        self.assertIn("pending approval", self.awaiting_page)


class SelectionCanonicalTests(MultiStepHarness):
    def test_selection_order_never_depends_on_request_order(self) -> None:
        config = self.config(require_approval=True)
        first = resolve_execution_selection_v3(
            config, planner_profile_id="planner",
            step_profile_ids={"S03": "impl-a", "S01": "impl-b", "S02": "impl-a"},
            reviewer_profile_id="reviewer",
        )
        second = resolve_execution_selection_v3(
            config, planner_profile_id="planner",
            step_profile_ids={"S01": "impl-b", "S02": "impl-a", "S03": "impl-a"},
            reviewer_profile_id="reviewer",
        )
        self.assertEqual(first, second)
        self.assertEqual([item.step_id for item in first.steps], ["S01", "S02", "S03"])
        ensure_execution_selection_v3(self.root / "sel", first)
        durable = json.loads((self.root / "sel" / "execution_selection.json").read_text())
        self.assertEqual([item["step_id"] for item in durable["steps"]], ["S01", "S02", "S03"])
        for invalid in ({"S01": "impl-a", "S03": "impl-a"}, {"S02": "impl-a"}, {"S07": "impl-a"}, {}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ExecutionSelectionError):
                    resolve_execution_selection_v3(
                        config, planner_profile_id="planner",
                        step_profile_ids=invalid, reviewer_profile_id="reviewer",
                    )


if __name__ == "__main__":
    unittest.main()
