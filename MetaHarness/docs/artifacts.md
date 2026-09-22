# Artifacts

Each run directory is the durable handoff point for an operator. Failed and
interrupted worktrees are intentionally retained for inspection. Every state
write goes through `RunStateStore` and uses atomic replacement.

## Run root

| Artifact | Contents |
| --- | --- |
| `spec.md` | Exact human SPEC copied at run creation |
| `context.txt` | Base-pinned planner context and locator warnings |
| `repository_reference.json` | Staging remote name, optional GitHub web URL, base SHA and immutable base URL |
| `run_options.json` | Frozen run options: budgets, semantic revision switch, repair-scope policy, selected profiles |
| `planner.request.txt` / `planner.raw.md` / `planner.usage.json` | Exact planner request (written before the call), raw answer (written before parsing) and token counters |
| `task_plan.json` / `task_plan_v2.json` | Parsed plan metadata and preserved raw plan |
| `implementation_contract.md` | Canonical plan summary rendered from the READY plan |
| `implementation_bundle.json` | Step IDs, recommended profiles and the SHA-256 of every step contract |
| `steps/Sxx/contract.md` | The single authoritative step contract, approved and executed byte-for-byte |
| `execution_selection.json` | Selected profile (role, driver, provider, model, effort, fingerprint) of every role and step |
| `check_authority.json` | Trusted check IDs and argv frozen at approval |
| `plan_approval.json` | One exclusive APPROVE/REJECT decision bound to the plan, bundle and execution-selection hashes |
| `planner.conversation.json` | Only when the driver officially returned a planner conversation handle (never simulated) |
| `planner_recovery.json` | Operator plan recovery: `source: operator`, previous and replacement raw hashes, `planner_called: false` |
| `setup/results.json` / `setup/*.log` | Workspace dependency setup results and redacted logs |
| `accepted-chain.json` | Every commit MetaHarness accepted on the run branch (commit, tree, parent) |
| `resume_checkpoint.json` | The next operation that has not yet succeeded, with its expected HEAD/tree and correction bundle hash |
| `repair_task.md` / `repair_task.json` | `REVISE / HUMAN` only: route, summary, findings, required fixes, missing tests, branch, worktree, run id |
| `publish.json` | Successful publication: mode, target, remote, run branch, commit SHA, optional safe GitHub URL |
| `state.json` / `state.lock` | Atomic run state; the lock file serializes every state write and is never served |
| `trace/events.v1.jsonl` | Observation-only META TRACE v1 stream (see `docs/architecture.md`) |
| `diagnostics.md` | Bounded operator summary written at the end of a run |

## Cycles

A run is a sequence of cycles `cycles/001`, `cycles/002`, ... Cycle 001 is
the initial implementation; every later cycle is one review-driven correction.

| Artifact | Contents |
| --- | --- |
| `cycles/NNN/cycle.json` | Cycle identity. For NNN > 001: the source review cycle, route, candidate SHA and SHA-256 of the accepted `review.json`, and the resulting kind |
| `cycles/NNN/implementation/steps/Sxx/` | Per-step prompt, events, report, `step.json` (trees, changed paths, usage), `diff.patch` and `token_diagnostics.json` |
| `cycles/NNN/checks/<stage>/` | One deterministic gate: `evidence.json`, `checks.json`, `checks/*.log`, `diff.patch`, `changed-files.txt` |
| `cycles/NNN/checks/<stage>/accepted.json` | The green tree accepted by that gate: commit, parent, tree, acceptance kind and exact mutable scope with its hash |
| `cycles/NNN/check-repair/<stage>/attempts/NNN/` | One bounded check-repair attempt: prompt, report, `attempt.json`, trees before/after and the scope it was allowed |
| `cycles/NNN/semantic-revision/` | Semantic revision or direct `REVISE / IMPLEMENTATION` correction: prompt, report, `pre_checks.json`, `tree_before.txt`, `tree_after.txt` |
| `cycles/NNN/correction/` | `REVISE / REPLAN` only: repair-planner request/answer, correction bundle and contracts, `execution_selection.json`, `scope_delta.json`, optional `scope_approval.json` |
| `cycles/NNN/candidate/commit.json` | Accepted candidate SHA/tree/parent plus the staging remote, run branch, verified remote SHA and `pushed_at` proof required before review |
| `cycles/NNN/review/` | Exact reviewer request and metadata, raw answer, normalized `review.json`, usage |
| `.../tree_after_failure.txt` | Tree left by a failed worker attempt; a resume restores the checkpoint tree (in-scope paths only) |
| `.../attempts/NN/` | Artifacts of a failed attempt, moved aside before a resumed retry of the same operation |

Gate stages are `post-implementation`, `post-semantic-revision`,
`post-review-implementation` and `post-review-replan`.

## Identity chain

```text
base SHA -> accepted step commits -> green gate acceptance
         -> candidate commit (pushed, remote tip verified) -> reviewer PASS
         -> publication of that exact SHA
```

Only green, accepted trees become commits; a red tree stays an artifact. A
reviewer is called only when the local candidate SHA, the remote run-branch
tip and the reviewed SHA are identical, and publication uses exactly the SHA
named by the durable reviewer PASS.

The state records references and bounded metadata, never API key values. The
values of the variables named by `api_key_env` are redacted from check logs,
agent artifacts and failure details; a staged diff containing one fails the
run with `SECRET_IN_DIFF` before any commit.
