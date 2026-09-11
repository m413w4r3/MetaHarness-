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
| `diff.patch` | Complete staged binary-capable diff |
| `changed-files.txt` | Staged path list |
| `checks.json` / `checks/` | Check metadata, full stdout, and stderr |
| `evidence.json` | Frozen tree SHA, checks, gate failures, and diff metadata |
| `reviewer.request.txt` | Exact reviewer user message (written before the call) |
| `reviewer.raw.md` / `review.json` | Raw review (written before parsing) and normalized verdict |
| `repair_task.md` / `repair_task.json` | REVISE only: route, summary, findings, required fixes, missing tests, branch, worktree, run id |
| `state.json` | Atomic run state and status transitions |

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
