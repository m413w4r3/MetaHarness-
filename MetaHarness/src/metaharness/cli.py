"""Command-line interface for MetaHarness."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    write_plan_approval,
)
from .config import ConfigError, load_config
from .agent.runtime import CodexRuntimeError, prepare_codex_home
from .execution_selection import is_profile_aware_run
from .gitops import GitError, assert_clean, git_root, resolve_commit
from .llm.chat import validate_endpoint
from .models import RunStatus
from .orchestrator import OrchestrationError, run_orchestrator
from .state import RunStateStore


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

    print(f"run: {result.run_dir}")
    print(f"status: {result.status.value}")
    if result.commit_sha:
        print(f"commit: {result.commit_sha}")
    if result.failure_reason:
        print(f"failure: {result.failure_reason}")
    if result.status is RunStatus.COMMITTED:
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
    if state.get("failure"):
        print("failure: " + json.dumps(state["failure"], ensure_ascii=False))
    if state.get("commit_sha"):
        print(f"commit: {state['commit_sha']}")
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
            if path.is_file() and path.name != "state.json"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


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


def _doctor(config_path: Path) -> int:
    """Check local prerequisites only; this function never constructs an LLM client."""

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    problems: list[str] = []
    config_dir = config_path.expanduser().resolve().parent
    print(f"config: {config_path.expanduser().resolve()}")
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
        if name in config.runtime_environment:
            print(f"OK env {name}: set")
        else:
            problems.append(f"required env {name} is not set")
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
        except GitError as exc:
            problems.append(str(exc))
    for label, path in (("runs_root", config.runs_root), ("worktrees_root", config.worktrees_root)):
        if path.exists() and not path.is_dir():
            problems.append(f"{label} is not a directory: {path}")
        else:
            print(f"{label}: {path}")
    path_value = config.runtime_environment.get("PATH", "")
    if shutil.which("codex", path=path_value):
        print("OK codex binary: present")
    else:
        problems.append("codex binary is not resolvable")
    try:
        codex_home = prepare_codex_home(config)
        print(f"OK codex home: {codex_home}")
        print("OK codex MCP isolation: none configured")
    except CodexRuntimeError as exc:
        problems.append(str(exc))
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
    status = subparsers.add_parser("status", help="show a run status")
    status.add_argument("--run", required=True, type=Path)
    show = subparsers.add_parser("show", help="show run state and artifacts")
    show.add_argument("--run", required=True, type=Path)
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
    if args.command == "status":
        return _status(args.run)
    if args.command == "show":
        return _show(args.run)
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
