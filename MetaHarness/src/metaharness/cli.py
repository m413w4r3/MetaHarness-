"""Command-line interface for MetaHarness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .gitops import GitError, assert_clean, git_root, resolve_commit
from .models import RunStatus
from .orchestrator import OrchestrationError, run_orchestrator
from .state import RunStateStore


def _endpoint(base_url: str, endpoint_path: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


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


def _doctor(config_path: Path) -> int:
    """Check local prerequisites only; this function never constructs an LLM client."""

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    problems: list[str] = []
    print(f"config: {config_path.expanduser().resolve()}")
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
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    print("doctor: PASS")
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
    return 2  # pragma: no cover - argparse restricts commands


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
