#!/usr/bin/env python3
"""Manual real-provider smoke test for the planner boundary.

This runner intentionally does not use :class:`metaharness.planning.Planner`:
that class can perform format repair, while this smoke test measures the
natural compatibility of each real provider with the existing prompt and
parser.  It never creates a worktree, starts an agent, runs checks, or calls a
reviewer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metaharness.llm.chat import OpenAIChatTextClient, validate_endpoint  # noqa: E402
from metaharness.models import LLMEndpointConfig  # noqa: E402
from metaharness.planning import (  # noqa: E402
    PlanDecision,
    PlanParseError,
    TaskPlan,
    build_planner_prompt,
    parse_task_plan,
)


DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "manual-results"
SPEC_PATH = REPOSITORY_ROOT / "tests" / "manual_fixtures" / "planner_spec.md"
CONTEXT_PATH = REPOSITORY_ROOT / "tests" / "manual_fixtures" / "planner_context.txt"
API_KEY_ENV_BY_PROVIDER = {
    "chatgpt": "META_SMOKE_GPT_API_KEY",
    "gemini": "META_SMOKE_GEMINI_API_KEY",
}


class SmokeConfigurationError(ValueError):
    """The manual run cannot start because its local configuration is invalid."""


@dataclass(frozen=True)
class ProviderRun:
    name: str
    display_name: str
    config: LLMEndpointConfig


ClientFactory = Callable[[LLMEndpointConfig], OpenAIChatTextClient]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SmokeConfigurationError(f"required environment variable is missing: {name}")
    return value


def _optional_api_key_env(name: str) -> str | None:
    # The client receives the variable name, never the value.  Presence is
    # intentional here: if an operator exports an empty key, the existing
    # client will report that invalid credential during the attempt.
    return name if name in os.environ else None


def provider_run(name: str) -> ProviderRun:
    """Build exactly the endpoint configuration required by one provider."""

    if name == "chatgpt":
        config = LLMEndpointConfig(
            base_url=_required_env("META_SMOKE_GPT_BASE_URL"),
            endpoint_path="/v1/chat/completions",
            model="chatgpt-web",
            api_key_env=_optional_api_key_env("META_SMOKE_GPT_API_KEY"),
            timeout_seconds=420,
            retries=0,
            extra_body={"new_chat": True},
        )
        return ProviderRun("chatgpt", "ChatGPT UI", config)

    if name == "gemini":
        config = LLMEndpointConfig(
            base_url=_required_env("META_SMOKE_GEMINI_BASE_URL"),
            endpoint_path="/v1/stateless/chat/completions",
            model=_required_env("META_SMOKE_GEMINI_MODEL"),
            api_key_env=_optional_api_key_env("META_SMOKE_GEMINI_API_KEY"),
            timeout_seconds=420,
            retries=0,
            extra_body={},
        )
        return ProviderRun("gemini", "Gemini", config)

    raise SmokeConfigurationError(f"unknown provider: {name}")


def validate_repeat(value: str | int) -> int:
    try:
        repeat = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--repeat must be an integer from 1 to 10") from exc
    if not 1 <= repeat <= 10:
        raise argparse.ArgumentTypeError("--repeat must be between 1 and 10")
    return repeat


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("chatgpt", "gemini", "matrix"))
    parser.add_argument(
        "--repeat",
        type=validate_repeat,
        default=3,
        metavar="N",
        help="number of independent real generations per provider (1-10; default: 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        metavar="DIR",
        help="parent directory for planner-smoke-<timestamp> (default: manual-results)",
    )
    return parser.parse_args(argv)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with open(fd, "wb", closefd=True) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _safe_error_text(error: BaseException) -> str:
    """Keep parser diagnostics useful while never copying an API key."""

    message = str(error)
    for env_name in API_KEY_ENV_BY_PROVIDER.values():
        secret = os.environ.get(env_name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return " ".join(message.split())[:1000]


def _empty_counts(repeat: int) -> dict[str, int]:
    return {
        "attempts": repeat,
        "parsed": 0,
        "parse_failed": 0,
        "transport_failed": 0,
        "ready": 0,
        "blocked": 0,
    }


def _plan_json(plan: TaskPlan) -> dict[str, object]:
    normalized = asdict(plan)
    normalized["decision"] = plan.decision.value
    return normalized


def run_provider(
    provider: ProviderRun,
    *,
    repeat: int,
    run_dir: Path,
    spec_path: Path = SPEC_PATH,
    context_path: Path = CONTEXT_PATH,
    client_factory: ClientFactory = OpenAIChatTextClient,
) -> dict[str, int]:
    """Run independent attempts for one provider and persist their artifacts."""

    try:
        client = client_factory(provider.config)
    except Exception as exc:
        # Construction validates the endpoint URL and is a configuration
        # failure, not a provider generation attempt.
        raise SmokeConfigurationError(
            f"{provider.name} client configuration is invalid ({type(exc).__name__})"
        ) from None

    counts = _empty_counts(repeat)
    provider_dir = run_dir / provider.name
    for attempt_number in range(1, repeat + 1):
        attempt_dir = provider_dir / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Read these for each attempt so every attempt follows the same exact
        # exchange, including if an operator changes a fixture between runs.
        spec = spec_path.read_text(encoding="utf-8")
        context = context_path.read_text(encoding="utf-8")
        prompt = build_planner_prompt(spec, context)
        _write_text(attempt_dir / "request.txt", prompt)

        try:
            result = client.complete(prompt)
        except Exception as exc:
            counts["transport_failed"] += 1
            _write_json(
                attempt_dir / "result.json",
                {
                    "provider": provider.name,
                    "attempt": attempt_number,
                    "status": "TRANSPORT_FAILED",
                    "error_type": type(exc).__name__,
                },
            )
            continue

        raw = result if isinstance(result, str) else getattr(result, "text", None)
        if not isinstance(raw, str):
            # This is a client/protocol failure after the HTTP call.  Keep the
            # same no-secret transport-failure shape rather than manufacturing
            # a parser input from an invalid client result.
            counts["transport_failed"] += 1
            _write_json(
                attempt_dir / "result.json",
                {
                    "provider": provider.name,
                    "attempt": attempt_number,
                    "status": "TRANSPORT_FAILED",
                    "error_type": "InvalidClientResult",
                },
            )
            continue

        # The raw generated Markdown is persisted before parsing, exactly as
        # returned by the OpenAI-compatible client.
        _write_text(attempt_dir / "response.raw.md", raw)
        try:
            plan = parse_task_plan(raw)
        except PlanParseError as exc:
            counts["parse_failed"] += 1
            _write_text(attempt_dir / "parse-error.txt", _safe_error_text(exc))
            _write_json(
                attempt_dir / "result.json",
                {
                    "provider": provider.name,
                    "attempt": attempt_number,
                    "status": "PARSE_FAILED",
                    "error_type": type(exc).__name__,
                    "error": _safe_error_text(exc),
                },
            )
            continue

        counts["parsed"] += 1
        decision_key = "ready" if plan.decision is PlanDecision.READY else "blocked"
        counts[decision_key] += 1
        _write_json(attempt_dir / "parsed.json", _plan_json(plan))
        result_json: dict[str, object] = {
            "provider": provider.name,
            "attempt": attempt_number,
            "status": "PARSED",
            "decision": plan.decision.value,
        }
        model = getattr(result, "model", None)
        usage = getattr(result, "usage", None)
        if isinstance(model, str):
            result_json["model"] = model
        if isinstance(usage, dict):
            result_json["usage"] = usage
        _write_json(attempt_dir / "result.json", result_json)

    return counts


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def create_run_dir(output_root: Path, timestamp: str | None = None) -> Path:
    run_dir = output_root / f"planner-smoke-{timestamp or _timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_smoke(
    provider_names: Sequence[str],
    *,
    repeat: int,
    output_root: Path,
    client_factory: ClientFactory = OpenAIChatTextClient,
    timestamp: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Run configured providers and return the output directory and summary."""

    if not 1 <= repeat <= 10:
        raise SmokeConfigurationError("repeat must be between 1 and 10")
    providers = [provider_run(name) for name in provider_names]
    for provider in providers:
        try:
            validate_endpoint(provider.config.base_url, provider.config.endpoint_path)
        except Exception as exc:
            raise SmokeConfigurationError(
                f"{provider.name} endpoint configuration is invalid ({type(exc).__name__})"
            ) from None
    run_dir = create_run_dir(output_root, timestamp)
    summary: dict[str, object] = {"schema_version": 1, "providers": {}}
    provider_summary = summary["providers"]
    assert isinstance(provider_summary, dict)
    for provider in providers:
        provider_summary[provider.name] = run_provider(
            provider,
            repeat=repeat,
            run_dir=run_dir,
            client_factory=client_factory,
        )
    _write_json(run_dir / "summary.json", summary)
    return run_dir, summary


def _all_attempts_parsed(summary: dict[str, object]) -> bool:
    providers = summary.get("providers")
    if not isinstance(providers, dict) or not providers:
        return False
    return all(
        isinstance(counts, dict)
        and counts.get("parsed") == counts.get("attempts")
        and counts.get("transport_failed") == 0
        for counts in providers.values()
    )


def _print_summary(summary: dict[str, object], run_dir: Path) -> None:
    providers = summary["providers"]
    assert isinstance(providers, dict)
    labels = {"chatgpt": "ChatGPT UI", "gemini": "Gemini"}
    for name in ("chatgpt", "gemini"):
        counts = providers.get(name)
        if isinstance(counts, dict):
            print(f"{labels[name]:<11}: {counts['parsed']}/{counts['attempts']} plans parsed")
    print(f"Artifacts: {run_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    provider_names = ("chatgpt", "gemini") if args.provider == "matrix" else (args.provider,)
    try:
        run_dir, summary = run_smoke(
            provider_names,
            repeat=args.repeat,
            output_root=args.output,
        )
    except SmokeConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Configuration or artifact error ({type(exc).__name__})", file=sys.stderr)
        return 2

    _print_summary(summary, run_dir)
    return 0 if _all_attempts_parsed(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
