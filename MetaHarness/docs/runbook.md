# Runbook

## Prerequisites

- Python 3.12 or newer.
- A clean Git base when `require_clean_base = true`.
- A configured planner and reviewer endpoint implementing the contract in
  [providers.md](providers.md).
- Any API key supplied through the environment variable named by
  `api_key_env`; never put the key itself in TOML.

For the AutoWork profile, set the six `META_*` endpoint/model variables before
loading `examples/autowork.toml`. Because relative TOML paths are resolved
from the configuration file, copy this example to the MetaHarness checkout
root (or adjust its paths) before running it. The endpoint paths can select a
ChatGPT bridge or a WebAI-to-API Gemini bridge.

## Local checks

From the MetaHarness checkout:

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The test suite includes a local fake HTTP planner/reviewer and fake Codex
executable for a complete no-network orchestration path.

## Configure and run

```sh
cp examples/autowork.toml autowork.local.toml
python -m metaharness.cli config-check --config autowork.local.toml
python -m metaharness.cli doctor --config autowork.local.toml
python -m metaharness.cli run \
  --config autowork.local.toml \
  --spec examples/spec-example.md \
  --run-id example-001
```

Use `status` for a compact state view and `show` for state plus artifact names:

```sh
python -m metaharness.cli status --run ../MetaHarness-runs/example-001
python -m metaharness.cli show --run ../MetaHarness-runs/example-001
```

## Failure handling

`BLOCKED`, `REVISE`, `FAIL`, check failures, timeouts, mutations, stale HEAD,
empty/oversized diffs, and review-boundary changes do not commit. V0 stops and
leaves the run directory and worktree available for inspection; it does not
automatically repair or retry implementation work. Resolve the issue as an
operator, then start a new run ID. Remove an obsolete worktree only through
the normal Git worktree workflow after confirming it is no longer needed.

The locator is advisory and never supplies source truth: context is read from
the resolved base commit. Nested repository instruction files are loaded when
the locator identifies code in their scope. A fenced code block in a planner
or reviewer response is treated as data, not as metadata.
