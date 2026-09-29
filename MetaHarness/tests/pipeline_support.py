"""In-process harness for the generic pipeline-v2 state machine.

Git, the deterministic checks, the commit gates and every durable artifact
are real.  The planner is a scripted chat client and every worker role
(implementer, auditor) is a scripted executor registered under a test driver.
Nothing calls a network.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Mapping

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
    """A READY META PLAN v2; each step is ``(id, write path, title)``."""

    blocks = []
    for step_id, path, objective in steps:
        blocks.append(f"""BEGIN STEP {step_id}
TITLE: {objective}
EXECUTION_CLASS: MECHANICAL
DEPENDS_ON: NONE

CONTEXT
{objective} lives in {path}; keep the surrounding conventions.

READ_SET
- {path} :: current content

WRITE_SET
- {path}

CREATE_SET
NONE

DELETE_SET
NONE

INSTRUCTIONS
1. {objective}.

INTERFACES
NONE

EXAMPLES
NONE

TESTS
- {objective} is covered by the configured test.

PITFALLS
- Do not change paths outside the declared sets.

DONE_WHEN
- {path} holds the requested content.

VERIFY
- Run the configured test.

END STEP {step_id}
""")
    mode = "SINGLE" if len(steps) == 1 else "STAGED"
    return f"""META PLAN v2

STATUS: READY
TITLE: {title}
MILESTONE_ID: M01
MILESTONE_TITLE: {title}

OBJECTIVE
Implement the requested feature.

CONSTRAINTS
Keep the change local.

MILESTONE_GOAL
The requested feature exists and its checks pass.

PROJECT_REMAINDER
NONE

EXECUTION_MODE: {mode}
STEP_COUNT: {len(steps)}

{"".join(blocks)}
ACCEPTANCE
The feature file holds the requested content.

REQUIRED_CHECKS
- test

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
    return plan(*steps, title=title).replace("{implementer}", "worker")


def audit_report(
    status: str = "DONE", *, fixed: str = "none", refactored: str = "none",
    remaining: str = "none", risks: str = "none",
) -> str:
    """One complete META AUDIT v1 answer; the only audit completion contract."""

    if status != "DONE" and remaining == "none":
        remaining = "the remaining work named by the audit"
    return (
        "META AUDIT v1\n\nSTATUS\n" + status +
        "\n\nFIXED\n- " + fixed + "\n\nREFACTORED\n- " + refactored +
        "\n\nREMAINING\n- " + remaining + "\n\nRISKS\n- " + risks +
        "\nEND META AUDIT\n"
    )


def continuation_answer(
    decision: str, *, milestone: str = "NONE", remaining: str = "- none",
    question: str = "NONE", plan_text: str | None = None,
) -> str:
    """One compact META CONTINUE v1 fixture."""

    lines = [
        "META CONTINUE v1", "", "DECISION", decision, "", "SUMMARY",
        "The milestone decision is recorded.", "", "REMAINING", remaining,
        "", "NEXT_MILESTONE", milestone, "", "SPEC_QUESTION", question,
    ]
    if plan_text is not None:
        lines += ["", "BEGIN NEXT PLAN", plan_text.rstrip(), "END NEXT PLAN"]
    return "\n".join(lines + ["", "END META CONTINUE", ""])


def audit(
    status: str = "DONE", *, fixed: str = "none", refactored: str = "none",
    remaining: str = "none", risks: str = "none",
) -> Script:
    """A scripted auditor worker answering one META AUDIT v1 report."""

    report = audit_report(
        status, fixed=fixed, refactored=refactored, remaining=remaining, risks=risks,
    )

    def answer(_request: AgentRunRequest) -> str:
        return report

    return answer


class ScriptedChat:
    """A chat client answering from a queue; the last answer repeats."""

    def __init__(self, answers: list[Any], *, name: str = "chat", events: list[dict[str, Any]] | None = None) -> None:
        self.answers = list(answers)
        self.requests: list[str] = []
        self.name = name
        self.events = events

    def complete(self, request: str) -> str:
        self.requests.append(request)
        if self.events is not None:
            self.events.append({"kind": f"{self.name} call", "request": request})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


class ScriptedPlannerMux:
    """Keep initial planner answers and continuation decisions independently scripted."""

    def __init__(self, planner: ScriptedChat, continuation: ScriptedChat) -> None:
        self.planner = planner
        self.continuation = continuation

    def complete(self, request: str) -> str:
        if "META CONTINUE v1" in request:
            return self.continuation.complete(request)
        return self.planner.complete(request)


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
        self.events: list[dict[str, Any]] = []

    def on(self, role: ExecutionRole, *scripts: Script) -> "ScriptedWorkers":
        self.scripts.setdefault(role, []).extend(scripts)
        return self

    def roles(self) -> list[str]:
        return [request.role.value for request in self.calls]

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.calls.append(request)
        self.events.append({
            "kind": f"{request.role.value} call",
            "role": request.role.value,
            "profile_id": request.profile_id,
            "prompt": request.prompt,
        })
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
    capabilities = AgentExecutorCapabilities(edits_workspace=True, reads_external_artifacts=True)
    driver = DRIVER
    driver_version = "test"

    def __init__(self, workers: ScriptedWorkers) -> None:
        self.workers = workers

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        return self.workers.run(request)


def ladder_ledger(
    harness: "PipelineHarness", *, cycle: int = 1, stage: str = "post-implementation",
) -> dict[str, Any]:
    """The durable red-gate recovery ladder of one gate episode."""

    path = (
        harness.run_dir() / "cycles" / f"{cycle:03d}" / "check-repair" / stage / "ladder.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def ladder_strategies(
    harness: "PipelineHarness", *, cycle: int = 1, stage: str = "post-implementation",
) -> list[str]:
    """The distinct ladder rungs one red gate consumed, in order."""

    return [entry["strategy"] for entry in ladder_ledger(harness, cycle=cycle, stage=stage)["entries"]]


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
        self.remote = self.root / "remote.git"
        self.remote.mkdir()
        git(self.remote, "init", "--bare", "-q")
        git(self.repo, "remote", "add", "origin", str(self.remote))
        git(self.repo, "push", "-q", "-u", "origin", "main")
        self.check = self.root / "check.py"
        # The gate is green on the base content and on the delivered one: the
        # baseline comparison only ever counts a *new* failure.
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip()"
            " in {'base', 'good'} else 1)\n",
            encoding="utf-8",
        )
        self.workers = ScriptedWorkers()
        self.events = self.workers.events
        register_executor_driver(
            DRIVER, lambda _profile, _runtime, **_kwargs: _Executor(self.workers), replace=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(
        self, *, scope_mode: str = "soft",
        publish: bool = False, github_pr: bool = False,
        extra_checks: str = "", per_step_gate: str | tuple[str, ...] = "",
        budget: Mapping[str, Any] | None = None,
    ) -> Any:
        path = self.root / "config.toml"
        self.config_path = path
        gate_ids = (per_step_gate,) if isinstance(per_step_gate, str) else tuple(per_step_gate)
        gate_ids = tuple(item for item in gate_ids if item)
        gate = ""
        if gate_ids:
            rendered = ", ".join(f'"{item}"' for item in gate_ids)
            gate = f"\n[gate]\nper_step = [{rendered}]\n"
        budget_section = ""
        if budget:
            entries = "".join(f"{key} = {value!r}\n" for key, value in budget.items())
            budget_section = f"\n[budget]\n{entries}"
        path.write_text(f"""
repo = {str(self.repo)!r}
base_ref = "main"
runs_root = {str(self.root / 'runs')!r}
worktrees_root = {str(self.root / 'worktrees')!r}
require_clean_base = true
{gate}
[planning]
protocol = "v2"
{budget_section}

[repository]
remote = "origin"
planner_remote_exploration = true

[approval]
require_plan_approval = false

[scope]
mode = {scope_mode!r}

[context]
always_files = []

[ui]
default_planner_profile = "planner"
default_audit_profile = "auditor"

[routing]
mechanical_profile = "worker"
reasoning_profile = "worker"
agentic_profile = "worker"

[publish]
enabled = {'true' if publish else 'false'}
remote = "origin"
mode = "run-branch"
{('[github]\nenabled = true\npull_request_mode = "create"' if github_pr else '')}

[model_profiles.planner]
display_name = "Planner"
roles = ["planner"]
driver = "openai-chat"
provider = "test"
model = "fake-planner"
selection_mode = "request"
base_url = "http://127.0.0.1:9"
endpoint_path = "/v1/chat/completions"

[model_profiles.worker]
display_name = "Worker"
roles = ["implementer"]
driver = "{DRIVER}"
provider = "test"
model = "fake-worker"
selection_mode = "cli"

[model_profiles.auditor]
display_name = "Auditor"
roles = ["auditor"]
driver = "{DRIVER}"
provider = "test"
model = "fake-auditor"
selection_mode = "cli"

[model_profiles.live_planner]
display_name = "Live Planner"
roles = ["planner"]
driver = "openai-chat"
provider = "live"
model = "live-planner"
selection_mode = "request"
base_url = "http://127.0.0.1:9"
endpoint_path = "/v1/chat/completions"

[model_profiles.live_auditor]
display_name = "Live Auditor"
roles = ["auditor"]
driver = "{DRIVER}"
provider = "live"
model = "live-auditor"
selection_mode = "cli"

[model_profiles.live_worker]
display_name = "Live Worker"
roles = ["implementer"]
driver = "{DRIVER}"
provider = "live"
model = "live-worker"
selection_mode = "cli"

[[check_catalog]]
id = "test"
argv = [{sys.executable!r}, {str(self.check)!r}]
timeout_seconds = 30

{extra_checks}""", encoding="utf-8")
        return load_config(path)

    def orchestrator(
        self, config: Any, *, planner: list[Any], auditor: list[Any] | None = None,
        continuation: list[Any] | None = None,
    ) -> Orchestrator:
        """Wire one run: a scripted planner and the scripted auditor queue.

        The deterministic gate answers with an audit whenever the candidate is
        not clean, so a run that reaches its gate needs an auditor script; the
        default script lets the audit accept the candidate unchanged.
        """

        self.planner = ScriptedChat(planner, name="planner", events=self.events)
        self.continuation = ScriptedChat(
            continuation or [continuation_answer("COMPLETE")],
            name="planner_continue", events=self.events,
        )
        self.auditor = auditor if auditor is not None else [audit()]
        self.workers.on(ExecutionRole.AUDITOR, *self.auditor)
        return Orchestrator(config, planner_client=ScriptedPlannerMux(self.planner, self.continuation))

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

    def trace_events(self, run_id: str = "run") -> list[dict[str, Any]]:
        """Return the production trace; tests assert order from this authority."""

        path = self.run_dir(run_id) / "trace" / "events.v1.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def trace_names(self, run_id: str = "run") -> list[str]:
        return [event["event"] for event in self.trace_events(run_id)]

    def remote_tip(self, branch: str) -> str:
        return git(self.remote, "rev-parse", f"refs/heads/{branch}")
