# Runbook

## Prerequisites

- Python 3.12 or newer.
- A clean Git base when `require_clean_base = true`.
- A configured planner and reviewer endpoint implementing the contract in
  [providers.md](providers.md).
- Any API key supplied through the environment variable named by
  `api_key_env`; never put the key itself in TOML.

`examples/autowork.toml` is usable in place: its relative paths are resolved
from `examples/`. The endpoint configuration is explicit; the secret is
loaded only from `Bridges/.env.models`. No `export` is needed.

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
metaharness config-check --config examples/autowork.toml
metaharness doctor --config examples/autowork.toml
metaharness run \
  --config examples/autowork.toml \
  --spec examples/spec-example.md \
  --run-id example-001
```

`doctor` never contacts a model. Besides local files, Git and executables it
runs `codex sandbox -- /bin/true` with the managed `CODEX_HOME` (this detects
bubblewrap/AppArmor refusals) and, for OpenAI-chat profiles on `127.0.0.1` or
`localhost` only, an unauthenticated `GET <base_url>/health` expecting
`{"status": "ok"}`.

On first use, authenticate the managed Codex runtime once:

```sh
CODEX_HOME="$HOME/.local/share/metaharness/codex" codex login
metaharness doctor --config examples/autowork.toml
```

Run `doctor` after this login. It checks Codex authentication separately from
the local sandbox probe and fails closed when the managed home cannot be
verified. It never makes a model request and never starts a login flow.

MetaHarness never copies the personal `CODEX_HOME`, MCP configuration, or
credentials into that managed runtime.

When a `claude-code` reviser profile is configured, authenticate Claude Code
in its managed home through the CLI itself. MetaHarness does not assume an
authentication subcommand that is absent from the installed version:

```sh
CLAUDE_CONFIG_DIR="$HOME/.local/share/metaharness/claude" claude
metaharness doctor --config examples/autowork.toml
```

Doctor first probes `claude --help`, then follows the supported `claude auth
status` path when that subcommand is advertised. Claude receives an empty
managed MCP file and never inherits the personal Claude settings, hooks or
MCP configuration.

Reviser failures are classified as `CLAUDE_AUTH_FAILURE`, `CLAUDE_TIMEOUT`,
`CLAUDE_FAILED`, or `CLAUDE_COMMITTED`.

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

Start the local run UI:

```sh
metaharness web --config examples/autowork.toml --port 8765
```

Open `http://127.0.0.1:8765/`, click `NEW RUN`, enter the SPEC and click
`CREATE RUN`. The normal flow is:

```text
http://127.0.0.1:8765/
↓
NEW RUN → SPEC → CREATE RUN → planner → APPROVE → Codex → checks → review
```

The CLI remains available for automation and file-based runs:

```sh
metaharness run --config examples/autowork.toml --spec my-spec.md
```

With `[planning] protocol = "v2"`, the approval card shows the execution mode,
the step count and, for every step, the recommended implementer, a profile
dropdown and the exact `steps/Sxx/contract.md` bytes hashed in
`implementation_bundle.json` — the same bytes each fresh Codex process
receives. After approval each step card shows its status (✓ ✗ ▶ …), recent
events (messages and tool names, never tool arguments) and token usage; the
TOKEN USAGE table sums planner, Luna and reviewer tokens.

Open the created run, read the canonical plan, then approve or reject it and
observe Codex progress, checks and review. The UI never runs Codex or checks,
changes a plan or worktree, commits, or writes `state.json` directly.

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
`PLAN_APPROVAL_INVALID`, `WORKSPACE_SETUP_FAILED`, `WORKSPACE_SETUP_TIMEOUT`,
`WORKSPACE_SETUP_MUTATED`, `AGENT_NO_CHANGE`, `TOCTOU_FAILURE`, `GIT_FAILURE`,
`STEP_CONTRACT_DRIFT` (a step's READ/WRITE/DELETE path is missing or a CREATE
path already exists in the tree before Codex), `STEP_WRITE_SET_VIOLATION`
(Git shows a changed path outside the step's WRITE ∪ CREATE ∪ DELETE sets).
A v2 failure detail always starts with `step=Sxx` when a step failed.
V0 stops and
leaves the run directory and worktree available for inspection; it does not
automatically repair or retry implementation work. Resolve the issue as an
operator, then start a new run ID. Remove an obsolete worktree only through
the normal Git worktree workflow after confirming it is no longer needed.

The locator is advisory and never supplies source truth: context is read from
the resolved base commit. Nested repository instruction files are loaded when
the locator identifies code in their scope. A fenced code block in a planner
or reviewer response is treated as data, not as metadata.
