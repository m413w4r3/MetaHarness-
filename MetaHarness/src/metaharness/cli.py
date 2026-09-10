"""Command-line interface for MetaHarness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metaharness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_check = subparsers.add_parser("config-check", help="validate a TOML config")
    config_check.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config-check":
        return _config_check(args.config)
    return 2  # pragma: no cover - argparse restricts commands


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
