# Artifacts

Each run directory is the durable handoff point for an operator. Failed and
interrupted worktrees are intentionally retained for inspection.

| Artifact | Contents |
| --- | --- |
| `spec.md` | Exact human SPEC copied at run creation |
| `context.txt` | Base-pinned planner context and locator warnings |
| `planner.request.txt` | Exact planner user message |
| `planner.raw.md` | Exact raw planner response |
| `task_plan.json` | Parsed plan metadata and preserved raw plan |
| `agent.prompt.txt` | Plan-only implementer prompt |
| `agent.events.jsonl` | Bounded Codex event stream |
| `agent.final.md` / `agent.result.json` | Implementer report and protocol metadata |
| `diff.patch` | Complete staged binary-capable diff |
| `changed-files.txt` | Staged path list |
| `checks.json` / `checks/` | Check metadata, full stdout, and stderr |
| `evidence.json` | Frozen tree SHA, checks, gate failures, and diff metadata |
| `reviewer.request.txt` | Exact reviewer user message |
| `reviewer.raw.md` / `review.json` | Raw review and normalized verdict |
| `state.json` | Atomic run state and status transitions |

The state records references and bounded metadata, not API key values. Full
prompts and diffs are retained as named artifacts so `show` can remain a
bounded summary command. State writes use atomic replacement through
`RunStateStore`.

The critical identity chain is:

```text
base commit SHA -> staged evidence tree SHA -> rechecked tree SHA -> commit SHA
```

The final commit is created only when the reviewer returns `PASS` with route
`NONE`, all required deterministic gates pass, HEAD is still the base SHA, the
index tree SHA is unchanged, and there are no unstaged or untracked changes.
