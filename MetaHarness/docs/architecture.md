# Architecture

MetaHarness V0 is a single-task state machine. It creates one
planner decision, one Codex implementation run, an optional Claude Code
revision, one deterministic evidence snapshot, and one independent semantic
review. There is no repair loop or behavior mock in V0.

## Source-of-truth boundaries

| Layer | Meaning | Authority |
| --- | --- | --- |
| Human SPEC | Product intent and acceptance target | Human request |
| Planner raw plan | Forensic planner response and decision record | Planner |
| Parsed plan | Machine control metadata and source for the canonical contract | Harness control/rendering |
| Implementation contract | Canonical executor input rendered from the parsed READY plan | Planner decisions, mechanically rendered |
| Human plan approval | Explicit decision on the exact presented plan artifacts | APPROVE/REJECT |
| Codex | Executor of the plan in an isolated worktree | Cannot change the plan or commit |
| Claude Code | Semantic reviser/corrector after Luna | Isolated managed config; cannot run checks or commit |
| Deterministic gates | Checks, diff, HEAD, and mutation evidence | Mechanical evidence |
| Reviewer | Semantic critic of SPEC, PLAN, diff, and evidence | PASS/REVISE/FAIL decision |
| Git tree SHA | Identity of the exact candidate tree submitted to checks and review | Candidate commit and publication gates |

The raw plan is preserved as a forensic artifact and remains the planner's
decision record. The parsed READY plan is rendered into one canonical
implementation contract for the implementer; the raw plan, SPEC, and planner
preamble/postamble are not sent to that agent.

## Run flow

1. Resolve the configured base ref and require a clean base when configured.
2. Build planner context from that exact commit. The locator is advisory: its
   paths and ranges are validated, then the source is read from Git at the
   base SHA. Applicable nested `AGENTS.md`/`CLAUDE.md` files are loaded.
3. Ask the planner for one labeled text plan. A BLOCKED plan stops the run.
4. Render `implementation_contract.md` from the parsed READY plan and persist
   the SHA-256 identity of the exact `planner.raw.md` and
   `implementation_contract.md` bytes.
5. If enabled, set `AWAITING_PLAN_APPROVAL` and wait for one approval artifact.
   `APPROVE` allows the run to continue; `REJECT` ends it as
   `PLAN_REJECTED`. This gate is before worktree creation, so rejection creates
   no worktree, agent execution, or commit. A decision cannot approve another
   plan because both hashes must match the state identity.
6. Create one worktree at the resolved base SHA and give Codex that contract
   only.
   After Codex exits, any commit, branch switch, branch creation/deletion or
   worktree creation/removal fails the run (`AGENT_COMMITTED` or
   `AGENT_GIT_VIOLATION`); the worktree is preserved, never reset.
7. If selected, Claude Code reads the resulting worktree and corrects
   authorized files without shell commands; a Claude commit fails the run as
   `CLAUDE_COMMITTED`.
8. Run configured checks, snapshot the candidate tree before and after each
   check, stage once, and freeze the full diff plus the index tree SHA as
   evidence. A check that mutates the candidate (required or not) fails the
   run before review; an empty, oversized or secret-bearing diff too.
9. Give the reviewer the original SPEC, raw PLAN, context, diff, checks, and
   implementer report. The reviewer must return a coherent labeled verdict.
10. On `PASS` with a green deterministic gate, `authorize_commit` re-derives
   every precondition (READY plan, agent exit 0, gate, reviewer answer parsed
   again, HEAD and branch, index tree, unstaged/untracked state, candidate
   tree) and `commit_reviewed_tree` creates the single harness commit.

Codex, checks and the locator run through `procutil.run_bounded`: no shell,
own process group, file-backed stdin/stdout/stderr, hard deadline, and
termination of the whole group at the deadline and after exit, so a
background child cannot modify the candidate after its snapshot. A
descendant that creates its own session escapes this cleanup (no cgroups in
V0).

Commit objects are built from the exact candidate tree object
(`git commit-tree`), not from the index, and the branch is advanced with a
compare-and-swap `git update-ref HEAD <new> <base>`. Commit hooks therefore
cannot restage content, and a moved HEAD makes the update fail. In v2, this
candidate commit is created before semantic review; publication remains gated
by the final reviewer PASS.

Planner and reviewer answers are free Markdown. The parser is tolerant on
presentation (headings, bold labels, bracket/colon markers, one whole-answer
fence) but strict on control values: `STATUS`, `VERDICT` and `ROUTE` lines
must be exactly one known token, content of code fences is never metadata,
and duplicated or contradictory control values fail closed. A `PASS` also
requires `ROUTE: NONE`, a passed deterministic gate, explicit empty
`REQUIRED FIXES` and `MISSING TESTS`, and FINDINGS consisting only of
`NONE` or structured `MINOR`/`NIT` records. No unrecognized finding text or
blocking severity can authorize a commit.

The implementer does not receive the original SPEC or the planner's raw
response. This makes the parsed canonical contract the execution boundary,
while preserving the raw response for forensics. The reviewer receives both SPEC and PLAN so it can independently
check that the plan preserved the product intent and that the diff followed
the plan. It also receives the mechanical evidence, so semantic approval
cannot replace deterministic checks.

## META PLAN v2 with revision (two bounded cycles)

With `[planning] protocol = "v2"` and `[revision] enabled = true`, the run is:

```text
SPEC
→ indexer + repo-aware planner
→ human-approved STAGED bundle
→ Luna steps
→ Claude correction C01
→ final deterministic checks C01
→ immutable C01 candidate commit
→ push exact C01 run-branch candidate
→ GPT reviewer via bridge #1
   ├ PASS → publish approved C01 candidate
   └ REVISE → bounded C02 (repair planner + scope validation)
       → C02 Luna
       → C02 Claude correction
       → final deterministic checks C02
       → immutable C02 candidate commit
       → push exact C02 run-branch candidate
       → GPT reviewer via bridge #2
          ├ PASS → publish approved C02 candidate
          └ otherwise → STOP / operator
```

- Maximum automatic cycles = 2. Reviewer #1 `REVISE / IMPLEMENTATION` and
  `REVISE / REPLAN` both enter the single bounded C02 repair cycle. `HUMAN`
  remains operator-controlled. Any new semantic REVISE after C02 is
  `REPAIR_EXHAUSTED / HUMAN_REQUIRED`.
- C02 writes `repair/C02/scope_delta.json`, derived from parsed plan mutation
  sets. New paths are governed by durable `repair_scope_policy` and
  `repair_scope_max_added_paths` options. `auto-bounded` is recommended for
  AutoWork with a bound of 4; `require-approval` writes an exact-hash
  `scope_approval.json` and pauses without allowing path editing.
- `revision.enabled` is the only activation authority. It is cross-validated
  at load time: protocol v2, an explicit `ui.default_reviser_profile` using the
  `claude-code` driver, an explicit `ui.default_repair_profile` using the
  `codex` driver, `max_cycles = 2`. Without it no Claude process and no C02
  ever start, whatever profiles exist in the catalogue.
- Every new run then uses `execution_selection.json` schema 4 (planner, steps,
  reviser, repair implementer, reviewer), approved in the UI with all four
  families visible. There is no v3 fallback; `state.execution` is only a view.
- C01 and C02 steps run through the same `_execute_codex_step` primitive, with
  the same ordered gates and failure reasons (`STEP_CONTRACT_DRIFT`,
  `CODEX_AUTH_FAILURE`, `AGENT_COMMITTED`, `AGENT_GIT_VIOLATION`,
  `AGENT_TIMEOUT`, `AGENT_FAILED`, `AGENT_NO_CHANGE`,
  `STEP_WRITE_SET_VIOLATION`). The final report never drives a decision.
- Both reviewers go through `_run_v2_reviewer`, which passes the actual
  `deterministic_passed` to the gate payload, to `parse_review` and to the
  commit gate. A reviewer PASS on a red required check is an invalid answer
  (`REVIEWER_OUTPUT_INVALID`): no commit, no push, no C02.
- Deterministic checks are selected by planner IDs from the trusted check
  catalog; MetaHarness owns the commands and arguments, always requires the
  configured defaults, and runs configured preflights before expensive workers.
- Reviewer #2 receives the original approved plan and the C02 repair plan, the
  C01 and C02 Luna reports, the C01 and C02 Claude revisions, the scope delta,
  and the cycle history.
- Artifacts are per cycle: C01 in `steps/`, `revision/C01/`, `checks/C01/`,
  `review/C01/` (root copies for historical runs); C02 in `repair/C02/`,
  `revision/C02/`, `checks/C02/`, `review/C02/`. The API exposes them as
  `cycle_artifacts`; top-level aliases describe the final cycle.
- Each cycle creates and pushes its exact immutable candidate on the run branch
  before semantic review. Publication happens only after the final reviewer
  PASS and a green gate, using that already-pushed approved candidate: no
  force, no tag, no delete, never `base_ref`/`main`, and never an automatic
  merge of the run branch.

## Durable checkpoints and resume (P29)

`resume.py` defines `ResumePhase` (`initial_step`, `claude_c01`,
`final_checks_c01`, `candidate_commit_c01`, `candidate_push_c01`,
`reviewer_c01`, `repair_planner`, `scope_approval`, `repair_step`, `claude_c02`,
`final_checks_c02`, `candidate_commit_c02`, `candidate_push_c02`,
`reviewer_c02`, `publish`) and `ResumeCheckpoint(phase, cycle, step_id,
expected_head_sha, expected_tree_sha, execution_selection_sha256, plan_identity)`
(plus the C02
repair-bundle and scope-delta hashes once repair planning and scope validation
succeed, since the C02 bundle has no human approval). `resume_checkpoint.json`
is written atomically after
every durable transition and always names the next operation that has not
yet succeeded; an operation is never marked complete before its artifacts
are durable.

`Orchestrator.resume(run_id)` (CLI `metaharness resume`, web
`POST /runs/<id>/resume`) is fail-closed: it re-validates approval, plan
identity, execution selection hash, worktree, branch, HEAD, exact candidate
tree, base SHA and scope before claiming the run with a compare-and-set state
transition, then re-enters `_execute_v2` at the checkpoint. Nothing critical
lives only in memory: step records, Claude revisions, evidence and accepted
reviewer answers are read back from `steps/Sxx/step.json`,
`revision/Cxx/report.json`, `evidence.json` and `review/Cxx/`, and their tree
chain is verified. Durable pre-revision checks and final evidence for the
exact current tree are reused, and a reviewer answer already accepted for
that tree is re-parsed rather than requested again.

## Conversation policy

Planner and reviewers never share a logical conversation
(`planner_thread != reviewer_thread`): every reviewer call is a fresh
completion that receives its context through MetaHarness artifacts, so a
reviewer never judges its own planning. The same bridge process, browser
connection and external-ui model are reused. `LLMConversationHandle`
(`provider_id`, `conversation_id`) exists only when a driver officially
returns one — the OpenAI-compatible bridge client does not, and MetaHarness
never fabricates or scrapes one. When a handle exists it is persisted in
`planner.conversation.json` and the C02 repair planner may continue that
conversation (`complete_in_conversation`); this is the only allowed reuse. A
reviewer reporting the planner's handle is rejected as
`REVIEWER_OUTPUT_INVALID`.

All run-state writes go through `RunStateStore`, which replaces JSON files
atomically. The plan approval artifact is atomically published without
replacement, so a second decision fails. In v2, the candidate commit is
created after final deterministic checks and before semantic review; the exact
candidate tree SHA is verified again immediately before that commit. Final
publication still requires reviewer approval. A changed index, HEAD, or
worktree causes the candidate boundary to fail.
