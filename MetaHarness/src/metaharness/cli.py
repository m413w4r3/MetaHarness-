"""Command-line interface for MetaHarness."""

from __future__ import annotations

import argparse
import http.client
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Mapping

from .approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    write_plan_approval,
)
from .config import ConfigError, load_config
from .diagnostics import write_run_diagnostics
from .agent.auth import check_codex_authentication
from .agent.codex import build_agent_environment
from .agent.runtime import CodexRuntimeError, prepare_codex_home
from .claude.agent import (
    _MAX_REVISION_TURNS,
    _REVISION_TOOLS,
    build_claude_environment,
)
from .claude.auth import check_claude_authentication
from .claude.runtime import ClaudeRuntimeError, prepare_claude_home
from .execution_selection import is_profile_aware_run
from .gitops import (
    GitError,
    assert_clean,
    build_repository_reference,
    git_root,
    repository_remote_url,
    resolve_commit,
    validate_base_branch,
    validate_run_branch,
)
from .llm.chat import validate_endpoint
from .models import HarnessConfig, ProfileDriver, PublishMode, RunStatus
from .orchestrator import OrchestrationError, resume_run, run_orchestrator
from .profiles import profiles_for_config
from .redaction import config_secret_values, redact
from .result import RunResult
from .resume import ResumeError
from .resume import resume_info
from .state import STATE_LOCK_NAME, RunStateStore

_SANDBOX_PROBE_ARGV = ("sandbox", "--", "/bin/true")
_SANDBOX_PROBE_TIMEOUT_SECONDS = 20
_BRIDGE_HEALTH_TIMEOUT_SECONDS = 2
_BRIDGE_HEALTH_MAX_BYTES = 64 * 1024
_LOCAL_BRIDGE_HOSTS = frozenset({"127.0.0.1", "localhost"})
_DOCTOR_DETAIL_CHARS = 300
_CODEX_HELP_TIMEOUT_SECONDS = 20
_CLAUDE_HELP_TIMEOUT_SECONDS = 20


def _endpoint(base_url: str, endpoint_path: str) -> str:
    return validate_endpoint(base_url, endpoint_path)


def _config_check(config_path: Path) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"repo: {config.repo}")
    print(f"base_ref: {config.base_ref}")
    print(f"runs_root: {config.runs_root}")
    print(f"worktrees_root: {config.worktrees_root}")
    print(
        "planner: "
        f"{_endpoint(config.planner.base_url, config.planner.endpoint_path)} "
        f"model={config.planner.model}"
    )
    print(
        "reviewer: "
        f"{_endpoint(config.reviewer.base_url, config.reviewer.endpoint_path)} "
        f"model={config.reviewer.model}"
    )
    print(f"agent: model={config.agent.model} effort={config.agent.effort}")
    print(
        "plan approval: "
        f"{'required' if config.approval.require_plan_approval else 'disabled'} "
        f"(poll={config.approval.poll_interval_seconds:g}s)"
    )
    print(
        "publish: "
        f"{'enabled' if config.publish.enabled else 'disabled'} "
        f"(remote={config.publish.remote}, mode={config.publish.mode})"
    )
    locator = "enabled" if config.context.locator_argv else "disabled"
    print(f"context locator: {locator}")
    print("checks: " + (", ".join(check.name for check in config.checks) or "none"))
    return 0


def _run(config_path: Path, spec_path: Path, run_id: str | None) -> int:
    try:
        result = run_orchestrator(config_path, spec_path, run_id=run_id)
    except (ConfigError, OrchestrationError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _report_result(result)


def _resume(config_path: Path, run_id: str) -> int:
    """Resume one run at its durable checkpoint; never replays a phase."""

    try:
        result = resume_run(config_path, run_id)
    except (ConfigError, ResumeError, OrchestrationError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _report_result(result)


def _report_result(result: RunResult) -> int:
    print(f"run: {result.run_dir}")
    print(f"status: {result.status.value}")
    if result.commit_sha:
        print(f"commit: {result.commit_sha}")
    if result.failure_reason:
        print(f"failure: {result.failure_reason}")
    if result.status in {RunStatus.COMMITTED, RunStatus.PUBLISHED}:
        return 0
    if result.status is RunStatus.INTERRUPTED:
        return 130
    return 1


def _load_run_state(run_dir: Path) -> tuple[Path, dict[str, object]]:
    directory = run_dir.expanduser().resolve()
    state_path = directory / "state.json"
    return directory, RunStateStore(state_path).load()


def _status(run_dir: Path) -> int:
    try:
        directory, state = _load_run_state(run_dir)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load run {run_dir}: {exc}", file=sys.stderr)
        return 2
    print(f"run: {directory}")
    print(f"run_id: {state.get('run_id', '')}")
    print(f"status: {state.get('status', '')}")
    info = resume_info(directory, state)
    print(f"resumable: {'yes' if info.resumable else 'no'}")
    print(f"checkpoint: {info.phase or '—'}")
    print(f"resume: {info.label or '—'}")
    if info.reason:
        print(f"resume refusal: {info.reason}")
    if state.get("failure"):
        print("failure: " + json.dumps(state["failure"], ensure_ascii=False))
    if state.get("commit_sha"):
        print(f"commit: {state['commit_sha']}")
    publish = state.get("publish")
    if isinstance(publish, dict) and publish.get("status") == "pushed":
        print(f"remote: {publish.get('remote')}")
        print(f"branch: {publish.get('branch')}")
    return 0


def _show(run_dir: Path) -> int:
    try:
        directory, state = _load_run_state(run_dir)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load run {run_dir}: {exc}", file=sys.stderr)
        return 2
    # Show is intentionally a bounded summary.  Full prompts, diffs and logs
    # remain in their named artifacts instead of being dumped to a terminal.
    summary = {
        "run_dir": str(directory),
        "state": state,
        "artifacts": sorted(
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path.name not in {"state.json", STATE_LOCK_NAME}
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _diagnostics(config_path: Path, target: str, *, stdout: bool = False) -> int:
    """Regenerate the derived report for a run id or an explicit run dir."""

    try:
        config = load_config(config_path)
        candidate = Path(target).expanduser()
        if candidate.is_dir():
            run_dir = candidate
        elif candidate.is_absolute() or candidate.parent != Path("."):
            raise ValueError("target must be an existing run directory or a run id")
        else:
            run_dir = config.runs_root / target
        produced = write_run_diagnostics(config, run_dir)
        if stdout:
            sys.stdout.write(produced.read_text(encoding="utf-8"))
        else:
            print(f"diagnostics: {produced}")
        return 0
    except (ConfigError, OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _write_plan_decision(run_dir: Path, decision: ApprovalDecision) -> int:
    try:
        directory, state = _load_run_state(run_dir)
        if state.get("status") != RunStatus.AWAITING_PLAN_APPROVAL.value:
            raise ApprovalError("run must be awaiting_plan_approval")
        profile_aware = is_profile_aware_run(state)
        if decision is ApprovalDecision.APPROVE and profile_aware:
            # The CLI cannot choose execution profiles; a schema-v1 approval
            # would silently downgrade this run to unapproved defaults.
            raise ApprovalError(
                "this run is profile-aware: approve it through the profile-aware "
                "web UI (metaharness web) so the implementer and reviewer "
                "profiles are recorded; no approval was written"
            )
        state_identity = state.get("plan_identity")
        if not isinstance(state_identity, dict):
            raise ApprovalError("run state has no plan identity")
        try:
            expected_identity = PlanIdentity(
                raw_sha256=state_identity["raw_sha256"],
                contract_sha256=state_identity["contract_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApprovalError("run state has an invalid plan identity") from exc
        actual_identity = compute_plan_identity_from_run(directory)
        if profile_aware:
            # REJECT executes nothing: only the plan artifacts are compared.
            matches = (actual_identity.raw_sha256, actual_identity.contract_sha256) == (
                expected_identity.raw_sha256,
                expected_identity.contract_sha256,
            )
        else:
            matches = actual_identity == expected_identity
        if not matches:
            raise ApprovalError("plan artifacts do not match state.plan_identity")
        write_plan_approval(
            directory,
            decision=decision,
            identity=expected_identity,
            source="cli",
        )
    except (ApprovalError, OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"plan approval written: {directory / 'plan_approval.json'}")
    return 0


def _usable_secret_value(value: object) -> bool:
    """Same acceptance rule as the HTTP client for an ``Authorization`` key.

    A non-empty ``str`` whose characters are all printable ASCII without
    space (0x21..0x7e).  The value itself is never displayed.
    """

    return (
        isinstance(value, str)
        and bool(value)
        and all(0x21 <= ord(character) <= 0x7E for character in value)
    )


def _bounded_detail(text: str, secrets: Iterable[str]) -> str:
    return redact(" ".join(text.split()), secrets)[:_DOCTOR_DETAIL_CHARS]


def _probe_codex_sandbox(
    codex: str,
    environment: Mapping[str, str],
    codex_home: Path,
    secrets: tuple[str, ...],
) -> str | None:
    """Run ``codex sandbox -- /bin/true``; return None or a redacted detail.

    No model is contacted and nothing is persisted: the output is only used
    for one bounded diagnostic line.
    """

    try:
        result = subprocess.run(
            [codex, *_SANDBOX_PROBE_ARGV],
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_SANDBOX_PROBE_TIMEOUT_SECONDS,
            env=dict(environment),
            cwd=codex_home,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {_SANDBOX_PROBE_TIMEOUT_SECONDS}s"
    except (OSError, ValueError) as exc:
        return _bounded_detail(f"could not start: {type(exc).__name__}", secrets)
    if result.returncode != 0:
        output = (result.stderr or "") + " " + (result.stdout or "")
        return _bounded_detail(f"exit {result.returncode}: {output}", secrets)
    return None


def _probe_codex_cli(
    codex: str,
    environment: Mapping[str, str],
    codex_home: Path,
    secrets: tuple[str, ...],
) -> tuple[bool, str | None]:
    """Validate the production Codex parser/config path without a model call.

    ``--help`` is deliberately last. This exercises the real ``exec`` parser
    and strict managed config loading, while ``--ephemeral`` and the absence
    of a prompt/model keep the probe read-only and non-persistent.
    """

    argv = [
        codex,
        "exec",
        "--strict-config",
        "--ephemeral",
        "--sandbox",
        "workspace-write",
        "--help",
    ]
    try:
        result = subprocess.run(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_CODEX_HELP_TIMEOUT_SECONDS,
            env=dict(environment),
            cwd=codex_home,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {_CODEX_HELP_TIMEOUT_SECONDS}s"
    except (OSError, ValueError) as exc:
        return False, _bounded_detail(f"could not start: {type(exc).__name__}", secrets)
    if result.returncode != 0:
        output = (result.stderr or "") + " " + (result.stdout or "")
        return False, _bounded_detail(f"exit {result.returncode}: {output}", secrets)
    return True, None


def _probe_claude_capabilities(
    claude: str, environment: Mapping[str, str], claude_home: Path
) -> tuple[bool, str | None]:
    """Ask Claude's parser to validate the managed runtime argv.

    ``--help`` is deliberately last: Claude exits before a prompt/model call,
    while unknown options still produce a non-zero parser failure.  Help text
    is not used as a capability manifest because Claude does not promise to
    list every supported option there.
    """

    argv = [
        claude,
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--bare",
        "--restricted",
        "--tools",
        _REVISION_TOOLS,
        "--no-session-persistence",
        "--no-chrome",
        "--disable-slash-commands",
        "--max-turns",
        str(_MAX_REVISION_TURNS),
        "--model",
        "probe",
        "--effort",
        "medium",
        "--permission-mode",
        "acceptEdits",
        "--settings",
        str(claude_home / "settings.json"),
        "--strict-mcp-config",
        "--mcp-config",
        str(claude_home / "empty-mcp.json"),
        "--help",
    ]

    try:
        result = subprocess.run(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_CLAUDE_HELP_TIMEOUT_SECONDS,
            env=dict(environment),
            cwd=claude_home,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except (OSError, ValueError) as exc:
        return False, type(exc).__name__
    # A failing help command is unsupported even if its output happens to
    # mention every flag name.
    if result.returncode != 0:
        return False, f"exit {result.returncode}"
    return True, None


def _doctor_claude(config: HarnessConfig, path_value: str) -> list[str]:
    """Check the managed Claude runtime; return at most one Claude problem.

    Each check runs only when the previous one passed, so one root cause
    yields one diagnostic, and none of them is ever reported as a Codex one.
    """

    claude = shutil.which("claude", path=path_value)
    if not claude:
        return ["claude binary is not resolvable"]
    print("OK claude binary: present")
    try:
        claude_home = prepare_claude_home(config)
    except ClaudeRuntimeError as exc:
        return [str(exc)]
    print(f"OK claude config home: {claude_home}")
    print("OK claude MCP isolation: empty")
    claude_environment = build_claude_environment(
        config.runtime_environment, claude_home=claude_home
    )
    supported, _detail = _probe_claude_capabilities(
        claude, claude_environment, claude_home
    )
    if not supported:
        return ["unsupported Claude Code CLI for MetaHarness reviser"]
    print("OK claude CLI capabilities: supported")
    auth_status = check_claude_authentication(
        claude_home, environment=claude_environment
    )
    if not auth_status.available:
        return [
            "Claude Code authentication unavailable\n"
            f"hint: authenticate the managed Claude runtime at {claude_home}"
        ]
    print("OK claude authentication: available")
    return []


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _probe_bridge_health(base_url: str) -> bool:
    """``GET <base_url>/health`` without credentials; expect ``{"status": "ok"}``."""

    url = f"{base_url.rstrip('/')}/health"
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    # No proxy, no redirect: the probe must reach exactly the local bridge.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    try:
        with opener.open(request, timeout=_BRIDGE_HEALTH_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return False
            body = response.read(_BRIDGE_HEALTH_MAX_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        return False
    if len(body) > _BRIDGE_HEALTH_MAX_BYTES:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def _local_bridge_urls(config: HarnessConfig) -> list[str]:
    """Base URLs of OpenAI-chat profiles served on this machine only."""

    urls: list[str] = []
    for profile in profiles_for_config(config).values():
        if profile.driver is not ProfileDriver.OPENAI_CHAT or not profile.base_url:
            continue
        try:
            hostname = urllib.parse.urlsplit(profile.base_url).hostname
        except ValueError:
            continue
        if hostname in _LOCAL_BRIDGE_HOSTS:
            normalized = profile.base_url.rstrip("/")
            if normalized not in urls:
                urls.append(normalized)
    return urls


def _doctor(config_path: Path) -> int:
    """Check local prerequisites only; this function never contacts a model.

    The only network access is an unauthenticated ``GET /health`` to a
    bridge configured on 127.0.0.1 or localhost.
    """

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    problems: list[str] = []
    secrets = config_secret_values(config, config.runtime_environment)
    config_dir = config_path.expanduser().resolve().parent
    print(f"config: {config_path.expanduser().resolve()}")
    git_executable = shutil.which("git", path=config.runtime_environment.get("PATH") or None)
    if git_executable:
        print("OK git executable: present")
    else:
        problems.append("git executable is not resolvable")
    for env_file in config.environment.files:
        try:
            label = str(env_file.relative_to(config_dir))
        except ValueError:
            label = str(env_file)
        print(f"OK environment file: {label}")
    required_env_names = {
        endpoint.api_key_env
        for endpoint in (config.planner, config.reviewer)
        if endpoint.api_key_env
    }
    required_env_names.update(
        profile.api_key_env
        for profile in config.model_profiles.values()
        if profile.api_key_env
    )
    for name in sorted(required_env_names):
        if _usable_secret_value(config.runtime_environment.get(name)):
            print(f"OK env {name}: usable")
        else:
            problems.append(f"required env {name} is missing or invalid")
    if not config.repo.is_dir():
        problems.append(f"repo does not exist: {config.repo}")
    else:
        try:
            repo = git_root(config.repo)
            print(f"git: {repo}")
            if config.require_clean_base:
                assert_clean(repo)
                print("clean base: PASS")
            base_sha = resolve_commit(repo, config.base_ref)
            print(f"base: {config.base_ref} ({base_sha})")
            if config.publish.enabled:
                fast_forward = config.publish.mode == PublishMode.FAST_FORWARD_BASE.value
                try:
                    repository_remote_url(repo, config.publish.remote)
                    validate_run_branch("harness/doctor/run", base_ref=config.base_ref)
                    if fast_forward:
                        # The fast-forward target must be a local branch with
                        # a remote-tracking ref; doctor never fetches.
                        validate_base_branch(repo, config.base_ref)
                        resolve_commit(repo, f"refs/heads/{config.base_ref}")
                        resolve_commit(
                            repo, f"refs/remotes/{config.publish.remote}/{config.base_ref}"
                        )
                except GitError as exc:
                    problems.append(f"publish preflight failed: {exc}")
                else:
                    print(f"OK publish remote: {config.publish.remote}")
                    print("OK publish branch namespace: harness/<plan>/<run-id>")
                    if fast_forward:
                        print(
                            f"OK publish target: {config.base_ref} via safe fast-forward "
                            f"({config.publish.remote}/{config.base_ref} tracked locally)"
                        )
            if config.repository.planner_remote_exploration and (
                config.repository_section_explicit or config.repository.web_url is not None
            ):
                try:
                    reference = build_repository_reference(
                        repo, base_sha=base_sha, config=config.repository
                    )
                except (GitError, ValueError) as exc:
                    problems.append(f"planner repository reference is invalid: {exc}")
                else:
                    if reference.web_url is None:
                        problems.append(
                            "planner repository remote URL is not a supported GitHub URL"
                        )
                    else:
                        print(f"OK planner repository: {reference.web_url}")
                    print(f"OK planner immutable base: {reference.base_sha}")
        except GitError as exc:
            problems.append(str(exc))
    for label, path in (("runs_root", config.runs_root), ("worktrees_root", config.worktrees_root)):
        if path.exists() and not path.is_dir():
            problems.append(f"{label} is not a directory: {path}")
        else:
            print(f"{label}: {path}")
    path_value = config.runtime_environment.get("PATH", "")
    codex = shutil.which("codex", path=path_value)
    if codex:
        print("OK codex binary: present")
    else:
        problems.append("codex binary is not resolvable")
    codex_home: Path | None = None
    try:
        codex_home = prepare_codex_home(config)
        print(f"OK codex home: {codex_home}")
        print("OK codex MCP isolation: none configured")
    except CodexRuntimeError as exc:
        problems.append(str(exc))
    if codex and codex_home is not None:
        # The environment Codex really receives: the agent allowlist taken
        # from the runtime mapping, API keys excluded, CODEX_HOME forced.
        probe_environment = build_agent_environment(
            config.agent,
            source_environment=config.runtime_environment,
            codex_home=codex_home,
            forbidden_names=required_env_names,
        )
        supported, _detail = _probe_codex_cli(
            codex, probe_environment, codex_home, secrets
        )
        if supported:
            print("OK codex CLI compatibility: supported")
        else:
            problems.append("unsupported Codex CLI for MetaHarness worker")
        detail = _probe_codex_sandbox(codex, probe_environment, codex_home, secrets)
        if detail is None:
            print("OK codex sandbox: usable")
        else:
            problems.append(f"codex sandbox probe failed\n  detail: {detail}")
        auth_status = check_codex_authentication(
            codex_home,
            environment=probe_environment,
        )
        if auth_status.available:
            print("OK codex authentication: available")
        elif auth_status.detail == "codex authentication is unavailable":
            problems.append(
                "codex authentication is unavailable for managed CODEX_HOME\n"
                f'hint: run CODEX_HOME="{codex_home}" codex login'
            )
        else:
            problems.append(
                "codex authentication could not be verified for managed CODEX_HOME\n"
                f'hint: run CODEX_HOME="{codex_home}" codex login'
            )
    if config.revision.enabled:
        print(
            "OK revision: enabled "
            f"(max_cycles={config.revision.max_cycles}, "
            f"reviser={config.ui.default_reviser_profile}, "
            f"repair={config.ui.default_repair_profile})"
        )
    else:
        print("OK revision: disabled")
    claude_profiles = tuple(
        profile for profile in profiles_for_config(config).values()
        if profile.driver is ProfileDriver.CLAUDE_CODE
    )
    if claude_profiles:
        problems.extend(_doctor_claude(config, path_value))
    for label, commands in (
        ("workspace setup", config.workspace_setup),
        ("check", config.checks),
    ):
        for command in commands:
            executable = command.argv[0] if command.argv else ""
            if shutil.which(executable, path=path_value):
                print(f"OK {label} executable {command.name}: resolvable")
            else:
                problems.append(f"{label} executable {command.name} is not resolvable")
    for base_url in _local_bridge_urls(config):
        if _probe_bridge_health(base_url):
            print("OK planner bridge: healthy")
        else:
            problems.append(f"planner bridge is not healthy: GET {base_url}/health")
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    print("doctor: PASS")
    return 0


def _web(config_path: Path, port: int) -> int:
    try:
        from .web.server import serve

        serve(config_path, port=port)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metaharness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_check = subparsers.add_parser("config-check", help="validate a TOML config")
    config_check.add_argument("--config", required=True, type=Path)
    run = subparsers.add_parser("run", help="execute one SPEC")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--spec", required=True, type=Path)
    run.add_argument("--run-id", type=str)
    resume = subparsers.add_parser(
        "resume", help="resume a failed run at its durable checkpoint"
    )
    resume.add_argument("--config", required=True, type=Path)
    resume.add_argument("--run-id", required=True, type=str)
    status = subparsers.add_parser("status", help="show a run status")
    status.add_argument("--run", required=True, type=Path)
    show = subparsers.add_parser("show", help="show run state and artifacts")
    show.add_argument("--run", required=True, type=Path)
    diagnostics = subparsers.add_parser("diagnostics", help="regenerate a run diagnostic report")
    diagnostics.add_argument("target", help="run id or run directory")
    diagnostics.add_argument("--config", required=True, type=Path)
    diagnostics.add_argument("--stdout", action="store_true", help="write the report to stdout")
    doctor = subparsers.add_parser("doctor", help="check local prerequisites")
    doctor.add_argument("--config", required=True, type=Path)
    approve = subparsers.add_parser(
        "approve-plan", help="approve the exact plan for a waiting run"
    )
    approve.add_argument("--run", required=True, type=Path)
    reject = subparsers.add_parser(
        "reject-plan", help="reject the exact plan for a waiting run"
    )
    reject.add_argument("--run", required=True, type=Path)
    web = subparsers.add_parser("web", help="serve the local observation UI")
    web.add_argument("--config", required=True, type=Path)
    web.add_argument("--port", default=8765, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config-check":
        return _config_check(args.config)
    if args.command == "run":
        return _run(args.config, args.spec, args.run_id)
    if args.command == "resume":
        return _resume(args.config, args.run_id)
    if args.command == "status":
        return _status(args.run)
    if args.command == "show":
        return _show(args.run)
    if args.command == "diagnostics":
        return _diagnostics(args.config, args.target, stdout=args.stdout)
    if args.command == "doctor":
        return _doctor(args.config)
    if args.command == "approve-plan":
        return _write_plan_decision(args.run, ApprovalDecision.APPROVE)
    if args.command == "reject-plan":
        return _write_plan_decision(args.run, ApprovalDecision.REJECT)
    if args.command == "web":
        return _web(args.config, args.port)
    return 2  # pragma: no cover - argparse restricts commands


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
