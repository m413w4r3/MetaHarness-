#!/usr/bin/env python3
"""Manual real-provider smoke test for the planner boundary.

It renders the production META PLAN v2 planner request for a fixed check
catalogue, sends it once, and applies the production v2
parser without any format repair.  It measures the natural compatibility of
each real provider with the real prompt and parser.  It never creates a
worktree, starts an agent, runs checks, or calls a reviewer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metaharness.llm.chat import OpenAIChatTextClient, validate_endpoint  # noqa: E402
from metaharness.models import (  # noqa: E402
    CheckConfig,
    LLMEndpointConfig,
    PlanDecision,
    TaskPlanV2,
)
from metaharness.planning_v2 import (  # noqa: E402
    PlanParseError,
    build_planner_payload_v2,
    parse_task_plan_v2,
)


DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "manual-results"
SPEC_PATH = REPOSITORY_ROOT / "tests" / "manual_fixtures" / "planner_spec.md"
CONTEXT_PATH = REPOSITORY_ROOT / "tests" / "manual_fixtures" / "planner_context.txt"
API_KEY_ENV_BY_PROVIDER = {
    "chatgpt": "META_SMOKE_GPT_API_KEY",
    "gemini": "META_SMOKE_GEMINI_API_KEY",
}
# Non-generative catalogue endpoint of WebAI-to-API used by ``preflight``.
GEMINI_MODELS_PATH = "/v1/stateless/models"
PREFLIGHT_TIMEOUT_SECONDS = 15
PREFLIGHT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SMOKE_CHECKS = (CheckConfig("test", ("python", "-m", "unittest")),)


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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "provider",
        choices=("chatgpt", "gemini", "matrix", "preflight"),
        help=(
            "provider to exercise; 'preflight' only checks variables, URLs, "
            "fixtures and the Gemini model catalogue, without generating any plan"
        ),
    )
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


_SENSITIVE_TEXT = (
    re.compile(r"(?i)\b(?:proxy-)?authorization\s*[:=]\s*[^\r\n,;]*"),
    re.compile(r"(?i)\b(?:set-)?cookie\s*[:=]\s*[^\r\n]*"),
    re.compile(r"(?i)\bbearer\s+\S+"),
)


def _safe_error_text(error: BaseException) -> str:
    """Return a bounded one-line diagnostic that never copies a credential.

    Only ``str(error)`` is used (never a traceback).  Configured API key
    values, Authorization/Cookie headers and bearer tokens are redacted
    before the text is truncated, so a cut can never expose a key prefix.
    """

    message = str(error)
    for env_name in API_KEY_ENV_BY_PROVIDER.values():
        secret = os.environ.get(env_name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    for pattern in _SENSITIVE_TEXT:
        message = pattern.sub("[REDACTED]", message)
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


def _plan_json(plan: TaskPlanV2) -> dict[str, object]:
    return {
        "decision": plan.decision.value,
        "title": plan.title,
        "execution_mode": getattr(plan.execution_mode, "value", plan.execution_mode),
        "step_ids": [step.id for step in plan.steps],
        "required_checks": list(plan.required_checks),
        "blockers": plan.blockers,
    }


def build_smoke_request(spec: str, context: str) -> str:
    """The production v2 planner request for the fixed smoke catalogue."""

    return build_planner_payload_v2(
        spec, context,
        check_catalog=SMOKE_CHECKS,
        default_check_ids=("test",),
    ).rendered


def parse_smoke_plan(raw: str) -> TaskPlanV2:
    return parse_task_plan_v2(
        raw,
        check_catalog=SMOKE_CHECKS,
        default_check_ids=("test",),
    )


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
        prompt = build_smoke_request(spec, context)
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
                    # Distinguishes 401, connection refused, timeout, bridge
                    # errors...; redacted and bounded, never a traceback.
                    "error": _safe_error_text(exc),
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
                    "error": "client result has no text",
                },
            )
            continue

        # The raw generated Markdown is persisted before parsing, exactly as
        # returned by the OpenAI-compatible client.
        _write_text(attempt_dir / "response.raw.md", raw)
        try:
            plan = parse_smoke_plan(raw)
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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the optional Authorization header to another location."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _api_key_header(env_name: str | None) -> dict[str, str]:
    if env_name is None:
        return {}
    value = os.environ.get(env_name, "")
    # Same rule as the chat client: a key http.client would reject (and quote
    # in its error) is a configuration error reported by name only.
    if not value or any(not 0x21 <= ord(character) <= 0x7E for character in value):
        raise SmokeConfigurationError(f"{env_name} contains an invalid value")
    return {"Authorization": f"Bearer {value}"}


def fetch_gemini_models(config: LLMEndpointConfig) -> list[str]:
    """Return the model IDs of the WebAI-to-API stateless catalogue (GET only)."""

    url = validate_endpoint(config.base_url, GEMINI_MODELS_PATH)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", **_api_key_header(config.api_key_env)},
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=PREFLIGHT_TIMEOUT_SECONDS) as response:
        body = response.read(PREFLIGHT_MAX_RESPONSE_BYTES + 1)
    if len(body) > PREFLIGHT_MAX_RESPONSE_BYTES:
        raise ValueError("model catalogue response is too large")
    payload = json.loads(body.decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("model catalogue has no data array")
    return [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]


ModelCatalogFetcher = Callable[[LLMEndpointConfig], list[str]]


def run_preflight(
    *,
    fixtures: Sequence[Path] = (SPEC_PATH, CONTEXT_PATH),
    fetch_models: ModelCatalogFetcher = fetch_gemini_models,
) -> tuple[int, list[str]]:
    """Check the local smoke configuration without generating any plan.

    Return ``(exit_code, report_lines)``: 0 when ready, 2 for an invalid
    local configuration, 1 when the Gemini catalogue probe fails or does not
    list ``META_SMOKE_GEMINI_MODEL``.  Credentials are reported by variable
    name only.
    """

    lines: list[str] = []
    configuration_ok = True
    probe_ok = True

    for path in fixtures:
        try:
            path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            configuration_ok = False
            lines.append(f"FAIL fixture {path}: unreadable ({type(exc).__name__})")
        else:
            lines.append(f"OK   fixture {path}")

    providers: dict[str, ProviderRun] = {}
    for name in ("chatgpt", "gemini"):
        try:
            provider = provider_run(name)
            validate_endpoint(provider.config.base_url, provider.config.endpoint_path)
            _api_key_header(provider.config.api_key_env)
        except SmokeConfigurationError as exc:
            configuration_ok = False
            lines.append(f"FAIL {name}: {exc}")
            continue
        except Exception as exc:
            configuration_ok = False
            lines.append(f"FAIL {name}: endpoint configuration is invalid ({_safe_error_text(exc)})")
            continue
        providers[name] = provider
        key_env = provider.config.api_key_env
        credential = f"{key_env} set" if key_env else f"{API_KEY_ENV_BY_PROVIDER[name]} not set"
        lines.append(
            f"OK   {name}: {provider.config.base_url}{provider.config.endpoint_path} "
            f"model={provider.config.model} ({credential})"
        )

    if "chatgpt" in providers:
        lines.append(
            "INFO chatgpt: no network probe; select the model manually in the "
            "ChatGPT UI before running the smoke test"
        )
    gemini = providers.get("gemini")
    if gemini is not None:
        try:
            models = fetch_models(gemini.config)
        except Exception as exc:
            probe_ok = False
            lines.append(f"FAIL gemini catalogue {GEMINI_MODELS_PATH}: {_safe_error_text(exc)}")
        else:
            if gemini.config.model in models:
                lines.append(f"OK   gemini catalogue lists {gemini.config.model}")
            else:
                probe_ok = False
                listed = ", ".join(models[:20]) or "none"
                lines.append(
                    f"FAIL gemini catalogue does not list {gemini.config.model} "
                    f"(available: {_safe_error_text(ValueError(listed))})"
                )

    if not configuration_ok:
        return 2, lines
    return (0 if probe_ok else 1), lines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.provider == "preflight":
        code, lines = run_preflight()
        for line in lines:
            print(line)
        print("Preflight: " + {0: "ready", 1: "provider probe failed", 2: "configuration invalid"}[code])
        return code
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
