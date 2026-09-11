# Artifacts

Each run directory is the durable handoff point for an operator. Failed and
interrupted worktrees are intentionally retained for inspection.

| Artifact | Contents |
| --- | --- |
| `spec.md` | Exact human SPEC copied at run creation |
| `context.txt` | Base-pinned planner context and locator warnings |
| `planner.request.txt` | Exact planner user message (written before the call) |
| `planner.raw.md` | Exact raw planner response (written before parsing) |
| `task_plan.json` | Parsed plan metadata and preserved raw plan |
| `implementation_contract.md` | Canonical contract rendered from the parsed READY plan |
| `plan_approval.json` | One exclusive APPROVE/REJECT decision bound to both plan SHA-256 hashes |
| `agent.prompt.txt` | Canonical-contract implementer prompt |
| `agent.events.jsonl` | Complete Codex stdout (JSONL events, parsed with bounded memory) |
| `agent.final.md` / `agent.result.json` | Implementer report and protocol metadata |
| `setup/results.json` / `setup/*.log` | Workspace dependency setup results and redacted logs |
| `diff.patch` | Complete staged binary-capable diff |
| `changed-files.txt` | Staged path list |
| `checks.json` / `checks/` | Check metadata, full stdout, and stderr |
| `evidence.json` | Frozen tree SHA, checks, gate failures, and diff metadata |
| `reviewer.request.txt` | Exact reviewer user message (written before the call) |
| `reviewer.raw.md` / `review.json` | Raw review (written before parsing) and normalized verdict |
| `repair_task.md` / `repair_task.json` | REVISE only: route, summary, findings, required fixes, missing tests, branch, worktree, run id |
| `state.json` | Atomic run state and status transitions |
| `state.lock` | Internal `flock` file serializing every state write; never served |
| `planner.usage.json` / `reviewer.usage.json` | Token counters: `input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, `output_tokens`, `reasoning_output_tokens`, `total_tokens` (absent counters are 0) |
| `implementation_bundle.json` | v2 only: step IDs, recommended profiles and the SHA-256 of every step contract |
| `steps/Sxx/contract.md` | v2 only: the single authoritative step contract, written at planning, approved and executed byte-for-byte |
| `steps/Sxx/agent.*` / `steps/Sxx/step.json` | v2 only: per-step Codex prompt, events, report, trees, changed paths and usage |

Historic P20 runs stored contracts as `steps/Sxx.contract.md`; the UI can
still display them, but new runs never write and never execute that layout.

The state records references and bounded metadata, not API key values. The
values of the variables named by `api_key_env` are redacted from check logs,
agent artifacts and failure details; a staged diff containing one fails the
run with `SECRET_IN_DIFF` before review and is persisted redacted. Full
prompts and diffs are retained as named artifacts so `show` can remain a
bounded summary command. State writes use atomic replacement through
`RunStateStore`. A REVISE never triggers automatic re-implementation.

When plan approval is enabled, `state.json` records `plan_identity` with the
lowercase SHA-256 hashes of the exact UTF-8 bytes in `planner.raw.md` and
`implementation_contract.md`. `plan_approval.json` must contain the same two
hashes. It is checked before the worktree exists, so a malformed artifact,
wrong hash, changed plan, or second decision fails closed. The `source` field
(`cli`, `web-ui`, or `test`) is informational and is not an authority signal.

The critical identity chain is:

```text
base commit SHA -> staged evidence tree SHA -> rechecked tree SHA -> commit SHA
```

The final commit is created only when the reviewer returns `PASS` with route
`NONE`, all required deterministic gates pass, HEAD is still the base SHA on
the run branch, the index tree SHA is unchanged, and there are no unstaged or
untracked changes. Its tree is the reviewed tree object itself
(`HEAD^{tree} == staged_tree_sha`).
