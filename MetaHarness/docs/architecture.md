# Architecture

MetaHarness V0 is a single-task, single-agent state machine. It creates one
planner decision, one Codex implementation run, one deterministic evidence
snapshot, and one independent semantic review. There is no multi-agent mode,
repair loop, or behavior mock in V0.

## Source-of-truth boundaries

| Layer | Meaning | Authority |
| --- | --- | --- |
| Human SPEC | Product intent and acceptance target | Human request |
| Planner raw plan | Forensic planner response and decision record | Planner |
| Parsed plan | Machine control metadata and source for the canonical contract | Harness control/rendering |
| Implementation contract | Canonical executor input rendered from the parsed READY plan | Planner decisions, mechanically rendered |
| Human plan approval | Explicit decision on the exact presented plan artifacts | APPROVE/REJECT |
| Codex | Executor of the plan in an isolated worktree | Cannot change the plan or commit |
| Deterministic gates | Checks, diff, HEAD, and mutation evidence | Mechanical evidence |
| Reviewer | Semantic critic of SPEC, PLAN, diff, and evidence | PASS/REVISE/FAIL decision |
| Git tree SHA | Identity of the reviewed staged code | Commit boundary |

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
7. Run configured checks, snapshot the candidate tree before and after each
   check, stage once, and freeze the full diff plus the index tree SHA as
   evidence. A check that mutates the candidate (required or not) fails the
   run before review; an empty, oversized or secret-bearing diff too.
8. Give the reviewer the original SPEC, raw PLAN, context, diff, checks, and
   implementer report. The reviewer must return a coherent labeled verdict.
9. On `PASS` with a green deterministic gate, `authorize_commit` re-derives
   every precondition (READY plan, agent exit 0, gate, reviewer answer parsed
   again, HEAD and branch, index tree, unstaged/untracked state, candidate
   tree) and `commit_reviewed_tree` creates the single harness commit.

Codex, checks and the locator run through `procutil.run_bounded`: no shell,
own process group, file-backed stdin/stdout/stderr, hard deadline, and
termination of the whole group at the deadline and after exit, so a
background child cannot modify the candidate after its snapshot. A
descendant that creates its own session escapes this cleanup (no cgroups in
V0).

The commit is built from the reviewed tree object (`git commit-tree`), not
from the index, and the branch is advanced with a compare-and-swap
`git update-ref HEAD <new> <base>`. Commit hooks therefore cannot restage
content, and a moved HEAD makes the update fail.

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

All run-state writes go through `RunStateStore`, which replaces JSON files
atomically. The plan approval artifact is atomically published without
replacement, so a second decision fails. No commit is created before review,
and the staged tree SHA is
verified again immediately before the commit. A changed index, HEAD, or
worktree causes the commit boundary to fail.
