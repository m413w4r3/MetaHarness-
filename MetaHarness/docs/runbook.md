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

## Plan approval

Set the following section to require the human gate:

```toml
[approval]
require_plan_approval = true
poll_interval_seconds = 0.5
```

Approval is manual and durable, and always happens before worktree creation:

```sh
python -m metaharness.cli status --run ../MetaHarness-runs/example-001
python -m metaharness.cli approve-plan --run ../MetaHarness-runs/example-001
# or, to terminate the run:
python -m metaharness.cli reject-plan --run ../MetaHarness-runs/example-001
```

Inspect `planner.raw.md`, `implementation_contract.md` and
`state.json.plan_identity` before deciding. The plan cannot be edited through
this primitive. A rejection ends the run as `PLAN_REJECTED`; no worktree,
agent, or commit is created. Ctrl-C while waiting is persisted as
`INTERRUPTED`.

Use `status` for a compact state view and `show` for state plus artifact names:

```sh
python -m metaharness.cli status --run ../MetaHarness-runs/example-001
python -m metaharness.cli show --run ../MetaHarness-runs/example-001
```

## Local web UI

Start the local observation and plan-approval UI in Terminal A:

```sh
metaharness web --config autowork.local.toml --port 8765
```

Run a SPEC in Terminal B:

```sh
metaharness run --config autowork.local.toml --spec my-spec.md
```

Open `http://127.0.0.1:8765/`, open the run, read the canonical plan, then
approve or reject it and observe Codex progress, checks and review. The UI is
read-only apart from the two plan-decision buttons: it never runs Codex or
checks, changes a plan or worktree, commits, or writes `state.json` directly.

The run page polls `GET /api/runs/<run_id>` every 2 seconds (status,
timeline, header, failure, approval buttons, plan, checks, review, reviewer
raw) and the Codex progress JSONL every second, so transitions appear without
a manual refresh. A JSONL event longer than 1 MiB is shown as
`[oversized Codex event omitted]`; the progress offset always moves forward.

Local security invariants: the server binds `127.0.0.1` only; every request
must carry `Host: 127.0.0.1:<port>` or `Host: localhost:<port>` exactly
(403 otherwise, against DNS rebinding); an approval `POST` needs the
in-memory token and, when an `Origin` header is present, the exact local
origin. The token is embedded only in a run page awaiting plan approval,
never in the run list. HTML responses carry a nonce-based CSP (no external
script, object, `<base>`, form target or framing), `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer` and `nosniff`. Artifact text is escaped
server-side and written with `textContent` client-side, never as HTML.

## Failure handling

`BLOCKED`, `PLAN_REJECTED`, `REVISE`, `FAIL`, check failures, timeouts, mutations, stale HEAD,
empty/oversized diffs, and review-boundary changes do not commit. Common
failure reasons in `state.json`: `PLANNER_OUTPUT_INVALID`,
`REVIEWER_OUTPUT_INVALID`, `LLM_FAILURE`, `AGENT_TIMEOUT`, `AGENT_FAILED`,
`AGENT_COMMITTED`, `AGENT_GIT_VIOLATION`, `CHECK_SETUP_INVALID`,
`CHECK_MUTATED`, `EMPTY_DIFF`, `DIFF_TOO_LARGE`, `SECRET_IN_DIFF`,
`DETERMINISTIC_GATE_FAILED`, `REVIEW_REVISE`, `REVIEW_FAIL`,
`PLAN_APPROVAL_INVALID`, `TOCTOU_FAILURE`, `GIT_FAILURE`. V0 stops and
leaves the run directory and worktree available for inspection; it does not
automatically repair or retry implementation work. Resolve the issue as an
operator, then start a new run ID. Remove an obsolete worktree only through
the normal Git worktree workflow after confirming it is no longer needed.

The locator is advisory and never supplies source truth: context is read from
the resolved base commit. Nested repository instruction files are loaded when
the locator identifies code in their scope. A fenced code block in a planner
or reviewer response is treated as data, not as metadata.
