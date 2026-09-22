"""In-process harness for the generic pipeline-v2 state machine.

Git, the deterministic checks, the commit gates and every durable artifact
are real.  The planner and reviewer are scripted chat clients and every
worker role (implementer, check-repair, semantic reviser) is a scripted
executor registered under a test driver.  Nothing calls a network.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from metaharness.agent import AgentExecutorCapabilities, AgentRunRequest, AgentRunResult, register_executor_driver
from metaharness.config import load_config
from metaharness.gitops import candidate_tree_sha
from metaharness.models import ExecutionRole
from metaharness.orchestrator import Orchestrator

DRIVER = "fake-worker"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def plan(*steps: tuple[str, str, str], title: str = "Add the feature") -> str:
    """A READY META PLAN v2; each step is ``(id, write path, objective)``."""

    blocks = []
    for step_id, path, objective in steps:
        blocks.append(f"""BEGIN STEP {step_id}
TITLE: {objective}
IMPLEMENTER_PROFILE: {{implementer}}
DEPENDS_ON: NONE

OBJECTIVE
{objective}

READ_SET
- {path} :: current content

WRITE_SET
- {path}

CREATE_SET
NONE

DELETE_SET
NONE

INSTRUCTIONS
1. {objective}

VERIFY
- Run the configured test.

FORBIDDEN
- Do not change paths outside the declared sets.

END STEP {step_id}
""")
    mode = "SINGLE" if len(steps) == 1 else "STAGED"
    return f"""META PLAN v2

STATUS: READY
TITLE: {title}

OBJECTIVE
Implement the requested feature.

CONSTRAINTS
Keep the change local.

EXECUTION_MODE: {mode}
STEP_COUNT: {len(steps)}
REVIEWER_PROFILE: reviewer

{"".join(blocks)}
ACCEPTANCE
The feature file holds the requested content.

TESTS
The configured test is the final evidence.

RISKS
NONE

BLOCKERS
NONE

END META PLAN
"""


def initial_plan(*steps: tuple[str, str, str], title: str = "Add the feature") -> str:
    return plan(*steps, title=title).replace("{implementer}", "worker")


def correction_plan(*steps: tuple[str, str, str], title: str = "Correct the feature") -> str:
    return plan(*steps, title=title).replace("{implementer}", "repairer")


def review(verdict: str = "PASS", route: str = "NONE") -> str:
    findings = "NONE" if verdict == "PASS" else "MINOR | the content needs a correction"
    fixes = "NONE" if verdict == "PASS" else "Fix feature.txt."
    return (
        f"VERDICT: {verdict}\nROUTE: {route}\nSUMMARY: scripted review\n"
        f"FINDINGS: {findings}\nREQUIRED FIXES: {fixes}\nMISSING TESTS: NONE\n"
        "RESIDUAL RISKS: NONE\n"
    )


class ScriptedChat:
    """A chat client answering from a queue; the last answer repeats."""

    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.requests: list[str] = []

    def complete(self, request: str) -> str:
        self.requests.append(request)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


Script = Callable[[AgentRunRequest], "str | AgentRunResult"]


def write(path: str, content: str, report: str = "done\n") -> Script:
    def action(request: AgentRunRequest) -> str:
        (request.worktree / path).write_text(content, encoding="utf-8")
        return report
    return action


class ScriptedWorkers:
    """Every worker role answers from its own queue of scripted actions."""

    def __init__(self) -> None:
        self.scripts: dict[ExecutionRole, list[Script]] = {}
        self.calls: list[AgentRunRequest] = []

    def on(self, role: ExecutionRole, *scripts: Script) -> "ScriptedWorkers":
        self.scripts.setdefault(role, []).extend(scripts)
        return self

    def roles(self) -> list[str]:
        return [request.role.value for request in self.calls]

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.calls.append(request)
        queue = self.scripts.get(request.role) or []
        if not queue:
            raise AssertionError(f"no scripted {request.role.value} worker left")
        action = queue.pop(0)
        before = candidate_tree_sha(request.worktree)
        request.artifact_dir.mkdir(parents=True, exist_ok=True)
        (request.artifact_dir / "agent.prompt.txt").write_text(request.prompt, encoding="utf-8")
        outcome = action(request)
        if isinstance(outcome, AgentRunResult):
            return outcome
        return AgentRunResult(
            status="completed", exit_reason=None, tree_before=before,
            tree_after=candidate_tree_sha(request.worktree), usage=None,
            external_session_id=None, report_path=None, exit_code=0,
            final_message=outcome, driver=DRIVER,
        )


class _Executor:
    capabilities = AgentExecutorCapabilities(edits_workspace=True)
    driver = DRIVER
    driver_version = "test"

    def __init__(self, workers: ScriptedWorkers) -> None:
        self.workers = workers

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        return self.workers.run(request)


class PipelineHarness(unittest.TestCase):
    """A real repository, a real run directory and scripted models."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.name", "MetaHarness pipeline tests")
        git(self.repo, "config", "user.email", "pipeline@example.invalid")
        (self.repo / "feature.txt").write_text("base\n", encoding="utf-8")
        (self.repo / "other.txt").write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "base")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.check = self.root / "check.py"
        # The gate is green only when feature.txt holds exactly "good".
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip() == 'good' else 1)\n",
            encoding="utf-8",
        )
        self.workers = ScriptedWorkers()
        register_executor_driver(
            DRIVER, lambda _profile, _runtime, **_kwargs: _Executor(self.workers), replace=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(
        self, *, check_repair: int = 0, review_repair: int = 0,
        semantic_revision: bool = False, scope_policy: str | None = None,
    ) -> Any:
        path = self.root / "config.toml"
        reviser = '\ndefault_reviser_profile = "reviser"' if semantic_revision or review_repair else ""
        repair = '\ndefault_repair_profile = "repairer"' if check_repair or review_repair else ""
        path.write_text(f"""
repo = {str(self.repo)!r}
base_ref = "main"
runs_root = {str(self.root / 'runs')!r}
worktrees_root = {str(self.root / 'worktrees')!r}
require_clean_base = true

[planning]
protocol = "v2"

[revision]
enabled = {'true' if semantic_revision else 'false'}
max_check_repair_attempts = {check_repair}
max_review_repair_cycles = {review_repair}

[approval]
require_plan_approval = false

[context]
always_files = []

[ui]
default_planner_profile = "planner"
default_implementer_profile = "worker"
default_reviewer_profile = "reviewer"{reviser}{repair}

[model_profiles.planner]
display_name = "Planner"
roles = ["planner"]
driver = "openai-chat"
provider = "test"
model = "fake-planner"
selection_mode = "request"
base_url = "http://127.0.0.1:9"
endpoint_path = "/v1/chat/completions"
retries = 0

[model_profiles.reviewer]
display_name = "Reviewer"
roles = ["reviewer"]
driver = "openai-chat"
provider = "test"
model = "fake-reviewer"
selection_mode = "request"
base_url = "http://127.0.0.1:9"
endpoint_path = "/v1/chat/completions"
retries = 0

[model_profiles.worker]
display_name = "Worker"
roles = ["implementer"]
driver = "{DRIVER}"
provider = "test"
model = "fake-worker"
selection_mode = "cli"

[model_profiles.repairer]
display_name = "Repairer"
roles = ["repair"]
driver = "{DRIVER}"
provider = "test"
model = "fake-repairer"
selection_mode = "cli"

[model_profiles.reviser]
display_name = "Reviser"
roles = ["reviser"]
driver = "{DRIVER}"
provider = "test"
model = "fake-reviser"
selection_mode = "cli"

[[checks]]
name = "test"
argv = [{sys.executable!r}, {str(self.check)!r}]
timeout_seconds = 30
""", encoding="utf-8")
        config = load_config(path)
        if scope_policy is not None:
            self.scope_policy = scope_policy
        return config

    def orchestrator(
        self, config: Any, *, planner: list[Any], reviewer: list[Any],
    ) -> Orchestrator:
        self.planner = ScriptedChat(planner)
        self.reviewer = ScriptedChat(reviewer)
        return Orchestrator(config, planner_client=self.planner, reviewer_client=self.reviewer)

    def run_dir(self, run_id: str = "run") -> Path:
        return self.root / "runs" / run_id

    def worktree(self, run_id: str = "run") -> Path:
        return self.root / "worktrees" / run_id

    def state(self, run_id: str = "run") -> dict[str, Any]:
        return json.loads((self.run_dir(run_id) / "state.json").read_text(encoding="utf-8"))

    def checkpoint(self, run_id: str = "run") -> dict[str, Any]:
        return json.loads(
            (self.run_dir(run_id) / "resume_checkpoint.json").read_text(encoding="utf-8")
        )
