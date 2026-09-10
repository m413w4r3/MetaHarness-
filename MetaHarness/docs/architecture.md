# Architecture

MetaHarness V0 is a single-task, single-agent state machine. It creates one
planner decision, one Codex implementation run, one deterministic evidence
snapshot, and one independent semantic review. There is no multi-agent mode,
repair loop, or behavior mock in V0.

## Source-of-truth boundaries

| Layer | Meaning | Authority |
| --- | --- | --- |
| Human SPEC | Product intent and acceptance target | Human request |
| Planner raw plan | Concrete implementation contract | Implementation authority |
| Parsed plan | Machine control metadata (READY/BLOCKED and fields) | Harness control only |
| Codex | Executor of the plan in an isolated worktree | Cannot change the plan or commit |
| Deterministic gates | Checks, diff, HEAD, and mutation evidence | Mechanical evidence |
| Reviewer | Semantic critic of SPEC, PLAN, diff, and evidence | PASS/REVISE/FAIL decision |
| Git tree SHA | Identity of the reviewed staged code | Commit boundary |

The raw plan is preserved as an artifact and is the exact text sent to the
implementer. The parsed plan is used for routing and metadata; it is not a
second implementation specification.

## Run flow

1. Resolve the configured base ref and require a clean base when configured.
2. Build planner context from that exact commit. The locator is advisory: its
   paths and ranges are validated, then the source is read from Git at the
   base SHA. Applicable nested `AGENTS.md`/`CLAUDE.md` files are loaded.
3. Ask the planner for one labeled text plan. A BLOCKED plan stops the run.
4. Create one worktree at the resolved base SHA and give Codex the plan only.
5. Run configured checks, snapshot mutations, stage once, and freeze the full
   diff plus the index tree SHA as evidence.
6. Give the reviewer the original SPEC, raw PLAN, context, diff, checks, and
   implementer report. The reviewer must return a coherent labeled verdict.
7. On `PASS` with a green deterministic gate, recheck HEAD, index tree SHA, and
   unstaged/untracked state, then create the single harness commit.

The implementer does not receive the original SPEC. This makes the planner's
raw plan the implementation authority, prevents an executor from silently
reinterpreting product intent, and keeps the planning/execution boundary
testable. The reviewer receives both SPEC and PLAN so it can independently
check that the plan preserved the product intent and that the diff followed
the plan. It also receives the mechanical evidence, so semantic approval
cannot replace deterministic checks.

All run-state writes go through `RunStateStore`, which replaces JSON files
atomically. No commit is created before review, and the staged tree SHA is
verified again immediately before the commit. A changed index, HEAD, or
worktree causes the commit boundary to fail.
