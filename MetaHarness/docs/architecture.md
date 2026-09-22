# Architecture

MetaHarness is a single-task state machine. It creates one planner
decision, profile-selected implementation and correction cycles, deterministic
evidence snapshots, and an independent final review. The executor/backend is
selected by each profile; business roles do not imply a particular vendor or
driver.

## Source-of-truth boundaries

| Layer | Meaning | Authority |
| --- | --- | --- |
| Human SPEC | Product intent and acceptance target | Human request |
| Planner raw plan | Forensic planner response and decision record | Planner |
| Parsed plan | Machine control metadata and source for the canonical contract | Harness control/rendering |
| Implementation contract | Canonical executor input rendered from the parsed READY plan | Planner decisions, mechanically rendered |
| Human plan approval | Explicit decision on the exact presented plan artifacts | APPROVE/REJECT |
| Execution profiles | Planner, implementer, correction and reviewer selections | Declared roles plus immutable runtime metadata |
| Agent executors/backends | Execute the selected profile in an isolated worktree | Cannot change the plan or commit |
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
6. Create one worktree at the resolved base SHA and give the selected
   implementation profile the canonical contract only. Each accepted step is
   committed by the harness after its gate; a failed attempt keeps durable
   tree/diff evidence and never enters the accepted chain.
7. Run the deterministic checks. A red result enters the bounded direct
   check-repair loop and is checked again; a green result proceeds to the
   semantic reviser, which compares the candidate with the SPEC and is itself
   followed by deterministic checks.
8. Freeze the accepted candidate tree and commit SHA, push that exact commit
   on the run branch, and give the immutable candidate to the final reviewer.
   `PASS` publishes that exact reviewed SHA. `REVISE / IMPLEMENTATION` routes
   to semantic correction, `REVISE / REPLAN` to the review repair planner and
   `REVISE / HUMAN` to the operator.
9. Every gate snapshots the candidate tree before and after each check,
   stages once, and preserves the full diff plus the index tree SHA as
   evidence. A check that mutates the candidate, an unexpected HEAD, a
   secret-bearing diff, or a tree mismatch fails closed before any repair
   agent is called.
10. The reviewer receives the original SPEC, plan, immutable candidate SHA,
    diff and checks. The reviewer must return a coherent labeled verdict;
    it does not receive or control the correction budgets.

The normative v2 sequence is:

```text
SPEC → planner → implementation steps → accepted commits
→ deterministic gate
   ├ FAIL → bounded direct check repair → deterministic gate
   └ PASS → semantic revision (when enabled) → deterministic gate
→ accepted candidate → exact candidate push → final reviewer
   ├ PASS → publish the exact reviewed SHA
   ├ IMPLEMENTATION → direct semantic correction
   ├ REPLAN → corrective planner → implementer
   └ HUMAN → operator
```

These are separate authorities: check repair is not semantic revision;
semantic revision is not the final reviewer; and `REVISE / IMPLEMENTATION` is
not `REVISE / REPLAN`. Every run freezes the selected profiles and budgets in
durable snapshots, so changing live defaults cannot change an existing run or
its resume path.

Selected executors, checks and the locator run through `procutil.run_bounded`:
no shell, own process group, file-backed stdin/stdout/stderr, hard deadline, and
termination of the whole group at the deadline and after exit, so a
background child cannot modify the candidate after its snapshot. A
descendant that creates its own session escapes this cleanup (no cgroups in
V0).

Commit objects are built from the exact candidate tree object
(`git commit-tree`), not from the index, and the branch is advanced with a
compare-and-swap `git update-ref HEAD <new> <base>`. Commit hooks therefore
cannot restage content, and a moved HEAD makes the update fail. In v2, the
accepted candidate is committed after semantic revision and its deterministic
checks, then pushed to the configured repository run branch before final
review; the remote tip must equal that candidate SHA. Publication remains
gated by the final reviewer PASS.

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

## META PLAN v2 pipeline

The normative v2 flow is documented in [pipeline-v2.md](pipeline-v2.md):

```text
PLAN → implementation step → accepted step commit → …
→ deterministic checks
   ├ FAIL → bounded direct check-repair loop → deterministic checks
   └ PASS → semantic revision → deterministic checks
→ accepted candidate → push run branch → final reviewer
   ├ PASS → publish the exact reviewed SHA
   ├ REVISE / IMPLEMENTATION → semantic correction
   ├ REVISE / REPLAN → review repair planner
   └ REVISE / HUMAN → operator
```

- Correction budgets are independent: `max_check_repair_attempts` bounds
  check-repair attempts and `max_review_repair_cycles` bounds review-driven
  correction cycles. `semantic_revision_enabled` only controls semantic
  revision. The reviewer never owns or receives those budgets.
- Check repair fixes a deterministic signal; semantic revision compares the
  immutable candidate to the SPEC; the final reviewer routes a candidate that
  has already passed the mechanical gates.
- A correction cycle writes `cycles/<number>/correction/scope_delta.json`, derived from parsed plan mutation
  sets. New paths are governed by durable `repair_scope_policy` and
  `repair_scope_max_added_paths` options. `auto-bounded` is recommended for
  AutoWork with a bound of 4; `require-approval` writes an exact-hash
  `scope_approval.json` and pauses without allowing path editing.
- Every new V2 run freezes its role-shaped execution authority in
  `execution_selection.json` schema 5: planner, per-step implementers,
  optional check-repair, optional semantic-reviser, and final reviewer.
  Driver/harness, provider, model and effort are independent profile fields;
  the profile's declared roles determine whether the selection is valid.
- Step capacity has a single syntax authority, `metaharness.step_ids`:
  `PROTOCOL_MAX_STEPS = 99` and the step IDs `S01..S99`. This is a protocol
  bound, not a recommended execution size. The parser, the implementation bundle, the
  execution selection (schema 3 and 4), resume checkpoints, usage accounting
  and the UI all import it; no module spells its own step bound. SINGLE is
  exactly one step, STAGED two to `MAX_STEPS`. A step contract is at most
  16000 characters; there is no aggregate contract-size limit. The planner
  target remains approximately 4000-6000 characters per step.
- Initial and correction-cycle steps run through the same generic step executor, with
  the same ordered gates and failure reasons (`STEP_CONTRACT_DRIFT`,
  `AGENT_AUTH_FAILURE`, `AGENT_GIT_VIOLATION`,
  `AGENT_TIMEOUT`, `AGENT_RUNTIME_FAILED`, `AGENT_NO_CHANGE`,
  `STEP_WRITE_SET_VIOLATION`). The final report never drives a decision.
- Both reviewers go through `_run_v2_reviewer`, which passes the actual
  `deterministic_passed` to the gate payload, to `parse_review` and to the
  commit gate. A reviewer PASS on a red required check is an invalid answer
  (`REVIEWER_OUTPUT_INVALID`): no commit, no push, no correction cycle.
- Deterministic checks are selected by planner IDs from the trusted check
  catalog; MetaHarness owns the commands and arguments, always requires the
  configured defaults, and runs configured preflights before expensive workers.
- The reviewer receives the original approved plan, the bounded repair
  evidence applicable to its route, and the cycle history.
- The pushed candidate commit is the reviewer's code authority. A reviewable
  candidate is always pushed before review, and the remote tip is verified to
  equal its SHA. When remote exploration is available, MetaHarness sends the
  BASE SHA, CANDIDATE SHA,
  immutable candidate URL, compare URL, changed paths, and deterministic
  evidence; it does not recopy the full diff into the reviewer prompt. The
  reviewer inspects the immutable candidate when it needs code, while a
  bounded diff fallback is used when durable push proof or an inspectable web
  URL is unavailable. URLs alone never authorize remote exploration.
  This keeps prompt size independent of the total diff size and step
  count.
- Every reviewable candidate creates and pushes its exact immutable tree on
  the run branch before the final reviewer, regardless of `publish.enabled`.
  Publication happens only after
  the final reviewer PASS and a green gate, using that already-pushed approved
  candidate: no force, no tag, no delete, never `base_ref`/`main`, and never
  an automatic merge of the run branch.
- `trace/events.v1.jsonl` records the workflow in order (`run`, `plan`, `step`,
  `checks`, `repair`, `revision`, `candidate.pushed`, `review`, `publish`).
  Session fields include driver, provider, model, effort, profile fingerprint,
  prompt bytes and tree identities whenever available.

## Durable checkpoints and resume (P29)

`resume.py` defines `ResumePhase` for context, planning, implementation,
deterministic gates, check repair, semantic revision, candidate push, final
review, correction planning and publication. A checkpoint always names the
next operation that has not succeeded and carries the exact expected HEAD and
tree. `ResumeCheckpoint(phase, cycle, step_id, expected_head_sha,
expected_tree_sha, execution_selection_sha256, plan_identity)` also carries
correction-bundle and scope-delta hashes once correction planning and scope
validation succeed. `resume_checkpoint.json` is written atomically after
every durable transition and always names the next operation that has not
yet succeeded; an operation is never marked complete before its artifacts
are durable.

`Orchestrator.resume(run_id)` (CLI `metaharness resume`, web
`POST /runs/<id>/resume`) is fail-closed: it re-validates approval, plan
identity, execution selection hash, worktree, branch, HEAD, exact candidate
tree, base SHA and scope before claiming the run with a compare-and-set state
transition, then re-enters `_execute_v2` at the checkpoint. Nothing critical
lives only in memory: step records, semantic revisions, evidence and accepted
reviewer answers are read back from `steps/Sxx/step.json`,
`semantic-revision/report.json`, `evidence.json` and `review/`, and their tree
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
`planner.conversation.json` purely as run history: no MetaHarness call
consumes it, and each correction planner is always a fresh completion. A
reviewer reporting the planner's handle is rejected as
`REVIEWER_OUTPUT_INVALID`.

## Compact correction planning

The pushed candidate commit is the code authority for correction planning.
The repair planner receives a compact index of the original plan — per step,
its ID, title, dependency, objective, approved mutation scope, VERIFY and
FORBIDDEN — never the full approved step contracts, and never worker
reports, token counters or tree SHAs. The full candidate diff is not inlined
when Git remote exploration is available: the planner gets the BASE SHA, the
CANDIDATE SHA, the immutable candidate URL, the `BASE...CANDIDATE`
compare URL, the changed paths, the diff byte size and the diff SHA256, and
inspects the immutable commit whenever it needs source-level evidence. A correction cycle
always starts in a fresh conversation; it never continues the initial
planner thread.

Transport has exactly three attempts, the client's configured
`retries = 2` plus the first try — no fourth attempt is ever added. The first
and second HTTP attempts send the same compact inline request. Only when the
retry loop actually reaches the third attempt, after two retryable HTTP
responses, does the request move its evidence into an attached
`repair-evidence.md` and keep a small control prompt inline. A non-retryable
status, a malformed 200, a parser failure or a network error never triggers
the file mode. `candidate.diff` is attached to that third attempt only when
remote exploration is unavailable; the full diff is never inline. Before any
transport, `cycles/<number>/correction/` durably records `planner.request.txt`,
`planner.request.fallback.txt`, `planner.evidence.md` and
`planner.request.meta.json`, so a transport failure stays diagnosable. The
attachment is a transport mode only: it carries data and grants no authority
over the META PLAN v2 protocol.

Initial planning keeps the SINGLE vs STAGED decomposition thresholds:
`planning.single_step_max_mutable_paths` decides when an initial task must be
decomposed, and it is never relaxed. A bounded correction step is instead
bounded by `planning.staged_step_max_mutable_paths`, the maximum a single worker
worker may already touch, for a SINGLE `S01` exactly as for a STAGED step. The
repair request states that one number itself, and the bound is per step, never
aggregate. An already produced repair planner raw response may be locally
revalidated on resume when its `planner.evidence.md` is byte-identical to the
current evidence packet; local recovery never calls the model, never deletes an
artifact, never rewrites the plan, and refuses any answer the strict parser or
the repair policy still rejects.

All run-state writes go through `RunStateStore`, which replaces JSON files
atomically. The plan approval artifact is atomically published without
replacement, so a second decision fails. In v2, the accepted candidate
commit is created after semantic revision and its final deterministic checks;
its exact tree SHA is verified again immediately before the commit and before
publication. A changed index, HEAD, or worktree causes the candidate boundary
to fail.

## Bounded direct check repair

The repair budget and the mutable scope are **independent decisions**. Each
red deterministic gate may consume one configured direct check-repair attempt
until `max_check_repair_attempts` is exhausted. Whether a repair expands the
scope is a separate question with its own, stricter answer.

Only soft failures such as `CHECK_FAILED:<check id>` enter this loop. A
`lint`, `typecheck` or `test` failure is retried inside the authorized scope;
the attempt tree and diff are durable evidence, and the deterministic gate is
run again. `CHECK_REPAIR_EXHAUSTED` is terminal after the configured bound.
The `deny-expansion` and `require-approval` policies govern scope growth
independently.

Expansion, when it happens, stays as strict as before. When an ordinary
`CHECK_FAILED:*` output names a path, MetaHarness treats it only as a
candidate: the path must exist in the exact Git tree, be tracked, and be a
test or fixture path. The run's `auto-bounded` policy and configured maximum
apply atomically; production, generated, untracked and ambiguous paths are
never added. Model output never decides the scope. Paths the first repair
already earned are carried into later attempts and are never lost, never
re-added, and never counted twice against the bound. Production paths remain
the responsibility of the approved plan and reviewer/correction process.

Paths already earned by a repair remain cumulative authority for later attempts, and are never re-added or counted twice. The durable scope artifact is checked on every resume; a divergent or malformed artifact is a `RESUME_INTEGRITY_FAILURE`.

The complete authority is always the union of the original plan scope and each validated repair scope:

```
original plan scope
  ∪ each validated repair scope
```

Any candidate path outside that union remains a `RESUME_INTEGRITY_FAILURE`.

Snapshots whose `pipeline` predates the repair-scope fields are not loaded.
Current runs use the `deny-expansion` default. An operator may explicitly opt in on
resume with `--allow-bounded-test-scope-expansion`; this writes the immutable,
write-once `repair_scope_override.json` artifact and can only select
`auto-bounded` with a bound from 1 through 100. The override is refused for
snapshots containing either current repair-scope field, and it does not alter
`run_options.json` or any other resume authority.

## Optional GitHub workstream metadata

The `[github]` integration is disabled by default and is implemented through
an injectable `GitHubWorkstreamClient`; the default null client performs no
network I/O. Issues and pull requests are metadata only. An issue's body is
external, untrusted context: it is never the SPEC, approved plan, run options,
or mutable scope, and the bootstrap does not inject it into model prompts.

Only `remote_branch`, `issue_number`, and `pull_request_number` are persisted
or emitted in workstream metadata. Tokens, authorization headers, credential
helper output, and secret environment values are never durable artifacts.
