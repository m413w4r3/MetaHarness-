# Refoundation v3 — baseline and authority map

Base commit: `45a71cf3a67ed880af43eb551fe5bc923d790451` (working tree clean).
Branch: `refactor/metaharness-v3` was **not** created in this environment — the
repository git directory is `/home/nill/perso/work/.git`, outside the sandbox
writable root, so `git switch -c` fails with `unable to create directory for
.git/refs/heads/refactor/metaharness-v3`. Nothing in this document changes
runtime behaviour.

Evidence rules: every statement below is derived from the tracked tree at that
commit. Symbol references were resolved with `git grep`; import edges were
resolved with an AST scan of `src/**/*.py` and `tests/**/*.py`; reachability was
computed over the module-level call graph of each module. No symbol is called
dead because of its name.

## Baseline metrics

| Metric | Value |
| --- | --- |
| Cross-module imports of a `_private` symbol (whole tree: `src`, `tests`, `scripts`, `examples`, `references`) | **101** |
| Of which in `orchestrator.py` | **50** (from 6 modules) |
| Of which in `orchestration/resume_validation.py` | **11** (from 2 modules) |
| Of which in `orchestration/check_repair.py` | **5** (from 1 module) |
| `_private` symbols *owned* by the eleven campaign modules and consumed outside | **3** (`_safe_run_id`, `_review_code_evidence`, `_load_completed_step`) |
| Historical/compatibility branches in the eleven modules | **33** = 32 in the ten source modules + 1 pinned in the state-machine test (enumerated per module) |
| `PROVEN_DEAD` candidates | **5** (see `DELETION CANDIDATES`) |

---

## 1. `src/metaharness/orchestrator.py` — 7788 lines, 119 methods on `Orchestrator`

**RESPONSIBILITY**
The façade and, today, the composition root plus every phase implementation that
has not been extracted: SPEC intake, run preparation (context, repository
reference, checkpoint), the `PipelineV2Operations` implementations (steps,
gates, check-repair, candidate, review, revision, publication), GitHub issue/PR
metadata, trace emission, diagnostics and exit projection. It also exposes the
module-level `run_orchestrator` / `resume_run` / `recover_plan_run`.

**PUBLIC AUTHORITY**
`Orchestrator` (`run`, `run_text`, `resume`, `recover_plan`), `run_orchestrator`,
`resume_run`, `recover_plan_run`, `__all__` (9 names). Re-exported by
`metaharness/__init__.py` (`Orchestrator`, `run_orchestrator`). Callers:
`cli.py`, `web/run_manager.py`, `web/api.py`, `remote/client.py`,
`remote/server.py`, and 14 modules under `tests/` (13 test modules plus
`tests/pipeline_support.py`).

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
- `_safe_run_id` — production consumer `src/metaharness/web/run_manager.py:9`.
- `_review_code_evidence` — test consumer `tests/test_review.py:13`.
- 50 private *inbound* imports (`orchestration/shared.py` 28,
  `orchestration/resume_validation.py` 8, `orchestration/revision.py` 6,
  `orchestration/check_repair.py` 4, `orchestration/candidate.py` 2,
  `orchestration/scope_repair.py` 2).

**HISTORICAL/COMPATIBILITY BRANCHES (5)**
1. `execute = run` (L1607) — dormant alias, zero call sites tree-wide.
2. `_migrate_historical_step_acceptance` (L2990), driven from `resume()` at
   L7171-7182 as `historical_commit_gate_stale_authority`.
3. `historical_check_repair_redaction_crash` (L7186-7188, `ATTRIBUTEERROR`).
4. `legacy_mismatch_sources=_archived_step_mismatches(...)` (L4560) feeding
   `_archived_step_mismatches` (L7684) — mismatch sources from archived attempts.
5. `contract_repair.legacy_prompt_bug_candidate/_proven/supersede_legacy_prompt_bug`
   (L4578-L4592) — legacy implementer prompt-bug supersede before a repair.

**TARGET OWNER AFTER REFOUNDATION**
Composition root only (invariant 8): run preparation, coordinator wiring,
`PipelineV2Operations` assembly, exit projection — target <= 500 lines
(invariant 10), no phase logic, no private cross-imports (invariant 7). Phase
implementations move to `orchestration/` services; trace/diagnostics stay in
`trace.py` / `diagnostics.py`.

---

## 2. `src/metaharness/planning_v2.py` — 3118 lines

**RESPONSIBILITY**
The META PLAN v2 protocol: strict parsing of the planner answer
(`parse_task_plan_v2`), step-contract rendering, the step-contract-repair
protocol (prompts, identity, output-invalid classifier, attempt files), the
implementation-bundle persistence, and the model drivers `PlannerV2`,
`RepairPlannerV2`, `StepContractRepairPlanner`.

**PUBLIC AUTHORITY**
`__all__` (56 names). The active surface consumed by the façade is the parser,
the three planners, `read_approved_step_contract`, `read_set_paths`,
`validate_*_policy`, `render_repair_plan_summary`, `render_repair_step_index`,
`persist_recovered_plan_artifacts`, `normalize_repair_step_id` (tests).

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
None. No other module imports a `_private` symbol of `planning_v2.py`.

**HISTORICAL/COMPATIBILITY BRANCHES (6)**
1. `persist_implementation_bundle = write_implementation_bundle` (L2314).
2. `render_profile_catalogue = render_safe_profile_catalogue` (L2315) — name
   collides with the active `recommendation.render_profile_catalogue`.
3. `persist_planning_artifacts_v2 = persist_planning_v2_artifacts` (L2344).
4. `_apply_execution_mode_policy` (L1785) — superseded by the `policy_parts`
   construction in `build_planner_payload_v2` (L1871-1888).
5. `_apply_decomposition_policy` (L1799) — same supersession.
6. `_TEXT_SECTIONS` (L81) — retired section table, no reader.

**TARGET OWNER AFTER REFOUNDATION**
Split by concern: `planning/parser.py` (pure, no I/O),
`planning/contracts.py` (render + persist), `planning/planners.py` (model
drivers). The aliases and the two superseded helpers are deleted with the v3
format (invariant 1); no file > 900 lines (invariant 9).

---

## 3. `src/metaharness/models.py` — 749 lines

**RESPONSIBILITY**
The frozen configuration/control vocabulary: enums (`RunStatus`, `GateStage`,
`CycleKind`, `ExecutionRole`, `ReviewVerdict`, `ReviewRoute`, `PlanDecision`,
`BlockerKind`, `ExecutionMode`, `ExecutionClass`) and validated frozen
dataclasses (`HarnessConfig`, `RoutingConfig`, `ModelProfile`,
`ImplementationStep`, `TaskPlanV2`, `RunCycle`, `ExecutionSelection`, ...).

**PUBLIC AUTHORITY**
No `__all__`; effectively the whole module. `HarnessConfig` is imported by 20
src modules, `RunStatus` by 15, `GateStage` by 12. It is the reference
vocabulary for state, checkpoints and run options.

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
None.

**HISTORICAL/COMPATIBILITY BRANCHES (2)**
1. `RunStatus.BLOCKED` (L239) — never written as a run status by any writer call
   in the tree (AST scan of `update/update_if_status/transition_if/initialize/
   write_checkpoint`); it survives only as a *read* value (compared in
   `orchestrator._diagnose_result`) and as the legacy state key
   `"blocked"` in `plan_recovery.RECOVERABLE_PLANNER_STATE_PAIRS`.
2. `UIConfig.default_implementer_profile` (L540) — the loader always sets it to
   `None` (`config.py:990-991`, "the former ui.default_implementer_profile is
   intentionally not consulted"); the dataclass field is only reachable from
   hand-built configs in tests.

**TARGET OWNER AFTER REFOUNDATION**
Stays the single model module; `RunStatus` remains the only durable status
vocabulary (invariant 3). If it grows, split by domain
(`models/config.py`, `models/pipeline.py`) but keep one enum owner.

---

## 4. `src/metaharness/run_options.py` — 473 lines

**RESPONSIBILITY**
The immutable, secret-free snapshot captured by every pipeline-v2 run: the
`RunOptions` dataclass, its canonical bytes and sha256, read/write of
`run_options.json`, and the derivation of an effective `HarnessConfig`.

**PUBLIC AUTHORITY**
`RunOptions`, `RunOptionsError`/`RunOptionsConflict`, `write_run_options`,
`read_run_options_for_state`, `read_run_options_with_sha256`(+`_and_raw`),
`run_options_sha256`, `canonical_run_options_bytes`, `effective_run_config`,
`effective_repair_scope_policy`. Consumers: `orchestrator.py`, `web/api.py`,
`pipeline_v2.py`, 9 test modules.

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
None.

**HISTORICAL/COMPATIBILITY BRANCHES (4)**
1. `default_implementer_profile` (L73-86) — "constructor-only migration aid.
   It is never serialized and is not read by the modern resolver."
2. `route_defaults` fallback onto `config.ui.default_implementer_profile`
   (L163-165) when the routing profiles are unknown.
3. `from_mapping` accepts a 5-key snapshot without `"recovery"` (L285-311).
4. `optional_recovery_fields` (L299-304) — snapshots frozen before output
   corrections or planner restarts existed.

**TARGET OWNER AFTER REFOUNDATION**
Single owner of the run-options schema. All legacy acceptance is concentrated in
`from_config`/`from_mapping` and deleted with v3 (invariant 1); the artifact
must stay one current format, readable-but-not-resumable for old runs
(invariant 2).

---

## 5. `src/metaharness/resume.py` — 642 lines

**RESPONSIBILITY**
The durable `resume_checkpoint.json` (schema 3): the `ResumeCheckpoint` model,
the `ResumePhase` vocabulary, the phase→status projections, the operation IDs,
and the read-only resume-eligibility projection `resume_info` (which decides
whether a run may be resumed, and under which operation).

**PUBLIC AUTHORITY**
`ResumePhase`, `write_checkpoint`/`read_checkpoint`/`read_checkpoint_record`,
`mark_checkpoint_completed`, `resume_info`/`ResumeInfo`/`resume_label`,
`plan_identity_from_mapping`, `pipeline_version_from_state`, the six operation
IDs, `ResumeError`/`ResumeNotAllowedError`/`ResumeIntegrityError`/
`ResumeRequiresOperatorError`. Consumers: `orchestrator.py` (the façade imports
18 names), `orchestration/recovery.py`, `pipeline_v2.py`, `web/api.py`,
`remote/*`.

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
None. (`_check_repair_exhaustion_info`, `_max_read_paths`, `_STRANDED_DETAIL`
stay module-local.)

**HISTORICAL/COMPATIBILITY BRANCHES (5)**
1. `STEP_ACCEPTANCE_INTEGRITY_OPERATION` (L256-258) — "a legacy stale-authority
   commit gate whose candidate evidence diverged".
2. `stranded_contract_repair` (L319-366) — "only the legacy projection of an
   invalid StepContractRepairPlanner answer onto `AGENT_CONTRACT_MISMATCH`".
3. `_check_repair_exhaustion_info` (L432-450) — recognizes the exhausted gate
   *or* "its exact legacy redaction crash" (`ATTRIBUTEERROR`).
4. `_RESUMABLE_STATUSES` (L246-249) — includes the legacy waiting statuses
   `waiting_contract_repair` / `waiting_check_repair`.
5. `planning_protocol != "v2"` refusal (L400-401) plus
   `pipeline_version_from_state` (L24-27) — old runs are readable, never
   resumable.

**TARGET OWNER AFTER REFOUNDATION**
Split `resume/checkpoint.py` (artifact + schema, write authority) from
`resume/eligibility.py` (read-only projection). Legacy shapes move into one
explicit `resume/legacy.py` that is read-only by construction (invariant 2).

---

## 6. `src/metaharness/recovery_policy.py` — 322 lines

**RESPONSIBILITY**
Deterministic classification of stable failure codes into a
`RecoveryDisposition` with budget semantics. Pure: no I/O, no clock, no model.
Its docstring states the rule that becomes invariant 6: "It never asks a model
to decide whether a failure is safe to recover from."

**PUBLIC AUTHORITY**
`classify_failure`, `RecoveryDecision`, `RecoveryDisposition`,
`RecoveryBudgets`, `ExecutionFallbacks`. Consumers: `models.py` (field types),
`orchestration/recovery.py`, `orchestrator.py`, `web/*`, tests.

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
None.

**HISTORICAL/COMPATIBILITY BRANCHES (0)**
None. The module-private code sets (`_HARD_STOP_CODES`, `_AUTH_CODES`,
`_TRANSIENT_AGENT_CODES`, `_TRANSIENT_EXTERNAL_CODES`,
`_CORRECTNESS_REPAIR_CODES`) are code tables, not version branches; every code in
them is still produced by the tree.

**TARGET OWNER AFTER REFOUNDATION**
Stays the single deterministic recovery authority (invariants 4, 5, 6). Nothing
outside this module may add a disposition, and no model output may select one.

---

## 7. `src/metaharness/orchestration/pipeline_v2.py` — 690 lines

**RESPONSIBILITY**
The generic pipeline-v2 state machine and the durable artifact layout. It owns
the order of the durable phases and the checkpoint written before each of them,
the cycle sequence (`001, 002, ...`), the gate episodes, the candidate/review/
publication order, the final gate selection, and the fail-closed
`PipelineFailure`.

**PUBLIC AUTHORITY**
`PipelineV2Coordinator.run(context, start, resumed)` is the only authority on
phase order; `PipelineV2Operations` is the explicit port through which every
side effect is injected; `PipelineV2Context`, `CyclePlan`, `PipelineFailure`,
`FailureDetail`, and the path helpers (`cycle_dir`, `step_dir`, `gate_dir`,
`check_repair_dir`, `review_dir`, `correction_dir`, `semantic_revision_dir`,
`cycle_record_path`, `gate_acceptance_path`, `check_repair_fingerprint`).

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
None.

**HISTORICAL/COMPATIBILITY BRANCHES (1)**
1. Tolerant stage inputs: `_stage_name` (L65) and `check_repair_fingerprint`
   (L46) accept `GateStage | str`, i.e. the legacy string form of a stage.

**TARGET OWNER AFTER REFOUNDATION**
Keeps the single authoritative state machine (invariant 3). The
`PipelineV2Operations` implementations currently living on `Orchestrator` move
to named phase services; the coordinator itself must stay free of I/O.

---

## 8. `src/metaharness/orchestration/check_repair.py` — 989 lines

**RESPONSIBILITY**
Bounded deterministic-gate repair: reading the failed-check logs, narrowing the
repair scope, rendering the check-repair prompt, the `CheckRepairCoordinator`
loop, and the `GateAcceptanceService` that records the accepted gate state.

**PUBLIC AUTHORITY**
`CheckRepairCoordinator`, `GateAcceptanceService`, `CheckRepairAttempt`,
`gate_mutable_authority`, `CheckRepairScope` (shared), plus the scope sources
`_HUMAN_SCOPE_SOURCE` / `_EVIDENCE_SCOPE_SOURCE` / `_SCOPE_REQUEST_SOURCE`.

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
- `_hard_failure_items` — `orchestration/resume_validation.py`.
- `_SCOPE_REQUEST_SOURCE`, `_check_repair_prompt`, `_hard_integrity_failures`,
  `_soft_check_failures` — `orchestrator.py` (4 of the façade's 50).

**HISTORICAL/COMPATIBILITY BRANCHES (2)**
1. Check-repair scope `schema_version == 2` (L511-523) — "preserve resume
   compatibility for prior attempts whose base scope was the entire approved
   cycle envelope".
2. Provenance alias `"auto-bounded failing-test evidence"` (L566) — "schema v2
   resume compatibility".

**TARGET OWNER AFTER REFOUNDATION**
The gate/check-repair domain service. Its cross-module private consumers become
public API of the module (invariant 7); the schema-2 branch moves to the
read-only legacy reader (invariant 2).

---

## 9. `src/metaharness/orchestration/recovery.py` — 440 lines

**RESPONSIBILITY**
Applies a `RecoveryDecision`: owns the durable recovery budgets, records every
consumed attempt with its tree boundary, emits the `recovery.*` trace, and
projects a disposition that left its loop onto exactly one durable run status.
It never runs a model, repairs a tree or touches an authority artifact.

**PUBLIC AUTHORITY**
`RecoveryCoordinator` (`budget_key`, `used`, `consume`, `record`, `admit`,
`trace`), `RecoveryAttempt`, `RecoveryAdmission`, `RecoveryTerminalState`,
`project_exit`, `terminal_state_for`, `normalize_exit_reason`, `failure_code`,
`MAX_RECOVERY_ATTEMPT_RECORDS`.

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
None.

**HISTORICAL/COMPATIBILITY BRANCHES (3)**
1. `_LEGACY_IDENTITY` (L167, L244-253) — attempt records written before stable
   `operation_id` identities existed; they are collapsed on re-record.
2. `_AUTH_ALIASES` (L44-47, L128-134) — provider-credential aliases collapsed
   onto one waiting condition.
3. `_CHECK_INFRA` / `_REMOTE` (L33-42) — legacy failure-code alias sets mapped
   onto `WAITING_CHECK_INFRASTRUCTURE` / `WAITING_REMOTE`.

**TARGET OWNER AFTER REFOUNDATION**
The single recovery authority. Invariant 4 is implemented here: a correctness
failure must spend its bounded automatic recovery inside this loop *before* any
`WAIT_HUMAN` projection.

---

## 10. `src/metaharness/orchestration/resume_validation.py` — 1298 lines

**RESPONSIBILITY**
The resume integrity boundary: every durable artifact reader is fail-closed and
returns `None` rather than a partially trusted value, and `validate_resume` is
the single gate in front of every resumed execution checkpoint. It is
cycle-agnostic and never calls a model, a check or a Git write.

**PUBLIC AUTHORITY**
`validate_resume`, `ResumedRun`, and the readers the façade consumes
(`candidate_evidence`, `completed_step_records`, `load_correction_plan`,
`read_candidate_record`, `read_cycle_record`, `verify_correction_scope`,
`validate_correction_bindings`).

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
- Consumed by the façade: `_accepted_review`, `_load_evidence`, `_load_revision`,
  `_persist_planner_conversation`, `_read_planner_conversation`,
  `_read_repository_reference`, `_reusable_pre_checks`,
  `_semantic_revision_scope` (8 of the façade's 50 imports) — and the façade
  re-exports `_accepted_review`, which `tests/test_pipeline_v2_machine.py:2645`
  patches through `metaharness.orchestrator`.
- Consumed by a test: `_load_completed_step`
  (`tests/test_pipeline_v2_invariants.py:45`).
- Inbound: `_hard_failure_items`, `gate_mutable_authority` from `check_repair`.

**HISTORICAL/COMPATIBILITY BRANCHES (4)**
1. `read_cycle_record` (L335) — cycle 001 records use schema 1, cycles >= 002
   use schema 2.
2. `_validate_gate_acceptance` (L460, L493-496) — accepts schema `{1, 2}`;
   `evidence_sha256` is required only for schema 2.
3. Semantic scope `authority.json` schema 1 (L693).
4. `validate_correction_bindings` (L395) — correction records require schema 2.

**TARGET OWNER AFTER REFOUNDATION**
One resume-integrity module, split into artifact readers and the validation gate.
Its 9 externally consumed private symbols are promoted to a named public API or
kept module-local (invariant 7); the façade stops re-exporting them, so no test
can patch a private symbol through the façade (invariant 12).

---

## 11. `tests/test_pipeline_v2_machine.py` — 2654 lines, 97 tests, 10 classes

**RESPONSIBILITY**
End-to-end behaviour of the generic pipeline-v2 state machine over a real Git
worktree with fake workers: single-cycle journeys, check-repair, review
correction cycles, scope approval, semantic revision, resume, Git-chain/trace,
resume authority, observation/publication authority.

**PUBLIC AUTHORITY**
None (test module). It is the global journey suite named by invariant 11.

**PRIVATE DEPENDENCIES USED OUTSIDE MODULE**
N/A — but it *consumes* two private façade surfaces, which invariant 12 forbids:
- `mock.patch("metaharness.orchestrator._accepted_review", ...)` (L2645), where
  `_accepted_review` is a private symbol of `resume_validation` re-exported by
  the façade;
- `orchestrator._reviewer_client = reviewer` (L613), a private instance
  attribute used as a test double.

**HISTORICAL/COMPATIBILITY BRANCHES (1)**
1. `test_historical_attributeerror_shape_resumes_the_same_gate_checkpoint`
   (L1109) — pins the legacy redaction-crash shape on purpose.

**TARGET OWNER AFTER REFOUNDATION**
Stays the journey suite (invariant 11). Local invariants move to module-local
test files; resume/publication journeys keep using only public façade entry
points (invariant 12).

---

## Invariants v3

1. Un seul format runtime courant.
2. Les anciens runs restent lisibles mais ne sont pas resumables.
3. Une seule machine d'état fait autorité.
4. Un failure de correctness déclenche une recovery autonome avant WAIT_HUMAN.
5. Les décisions security/integrity/authority restent déterministes et
   fail-closed.
6. Un modèle ne décide jamais de sa propre autorité.
7. Aucun module n'importe un `_private_symbol` d'un autre module.
8. `orchestrator.py` devient uniquement une façade/composition root.
9. Pas de nouveau fichier Python > 900 lignes.
10. Objectif final : `orchestrator.py` <= 500 lignes.
11. Les tests state-machine globaux couvrent des parcours ; les invariants
    locaux appartiennent aux modules locaux.
12. Aucun test ne doit dépendre d'un private symbol de la façade.

Current conformity at the base commit:

| Invariant | State | Evidence |
| --- | --- | --- |
| 1 | held | `pipeline_version == 2` enforced in `state.py:70`, `trace.py:98`, `resume.py:25`, `run_options.py:87`. |
| 2 | held | `resume_info` refuses non-`v2` runs (`resume.py:400`); old shapes are read-only. |
| 3 | partly | `PipelineV2Coordinator` owns phase order, but 120 façade methods implement the operations. |
| 4 | partly | `RecoveryCoordinator` bounds recovery, but `WAIT_HUMAN` projections exist beside it. |
| 5 | held | `recovery_policy.py` is pure; `resume_validation.py` is fail-closed. |
| 6 | held | No planner/reviewer output selects a disposition or an authority. |
| 7 | **violated** | 101 cross-module private imports (50 in the façade alone). |
| 8 | **violated** | `orchestrator.py` = 7788 lines, 119 methods. |
| 9 | n/a | No new file yet; 5 existing files already exceed 900 lines. |
| 10 | **violated** | 7788 > 500. |
| 11 | partly | Journey coverage exists; module-local invariants are spread across 50 test modules. |
| 12 | **violated** | `tests/test_review.py:13` imports `_review_code_evidence` from the façade; `tests/test_pipeline_v2_machine.py:2645` patches `metaharness.orchestrator._accepted_review`. |

---

## DELETION CANDIDATES

Reference counts are exact `git grep` results over the tracked tree
(`src` + `tests` + `docs`) at the base commit. "Importers" lists every module
that imports the symbol. Nothing below has been removed.

### D1. `references/agent_runner.py` (whole file, 989 lines)

- `git grep -In 'agent_runner'` -> **1** hit: `references/agent_runner.py:462`,
  a string inside a generated commit message produced by the script itself.
- Importers: **none**. An AST scan of `src/**/*.py` + `tests/**/*.py` finds no
  import of `references`, `agent_runner` or any name it defines.
- Entrypoint: `main()` (L951) with an `argparse` CLI and
  `if __name__ == "__main__"` (L989). It is **not** registered in
  `pyproject.toml [project.scripts]` (`metaharness = "metaharness.cli:main"` is
  the only entry point), and it is **not** packaged:
  `[tool.setuptools.packages.find] where = ["src"]`. No reference in `docs/`,
  `scripts/`, `README.md`, or in the CI workflow
  `.github/workflows/metaharness-ci.yml` (repository root), which runs
  `python -m unittest discover -s tests` and therefore never imports it.
- Verdict: **PROVEN_DEAD**. Unreachable from every tracked entry point; its only
  residual use would be a manual `python references/agent_runner.py`, which no
  tracked document instructs. Deletion is a product decision, not a code risk.

### D2. `src/metaharness/planning_v2.py::_TEXT_SECTIONS` (L81)

- `git grep -n '\b_TEXT_SECTIONS\b'` -> **1** hit, the definition.
- Importers: none. In-module references: **0**.
- Entrypoint: none; the module parses with `_ENVELOPE_INLINE`,
  `_ENVELOPE_SECTIONS`, `_STEP_INLINE`, `_STEP_SECTIONS`.
- Verdict: **PROVEN_DEAD**.

### D3. `src/metaharness/planning_v2.py::_apply_execution_mode_policy` (L1785)

- `git grep -n '\b_apply_execution_mode_policy\b'` -> **1** hit, the definition.
- Importers: none. In-module references: **0**. Superseded by the
  `policy_parts` construction inside `build_planner_payload_v2` (L1871-1888),
  which calls `render_require_staged_policy_text` directly.
- Verdict: **PROVEN_DEAD**.

### D4. `src/metaharness/planning_v2.py::_apply_decomposition_policy` (L1799)

- `git grep -n '\b_apply_decomposition_policy\b'` -> **1** hit, the definition.
- Importers: none. In-module references: **0**. Same supersession as D3.
- Verdict: **PROVEN_DEAD**.

### D5. `src/metaharness/orchestrator.py::Orchestrator.execute` (alias, L1607)

- `git grep -n '\.execute('` over `src` + `tests` -> **0** hits.
- Importers: none; the alias is not exported in `__all__`.
- Entrypoint: none. The same dormant pattern exists at
  `src/metaharness/agent/codex.py:338` (`CodexAgent.execute = run`, also 0 call
  sites) — reported here, evaluated with the same evidence.
- Verdict: **PROVEN_DEAD** (dormant alias; removing it changes no call path).

### D6. `src/metaharness/planning_v2.py` legacy public aliases and unreached API

`persist_implementation_bundle` (L2314), `render_profile_catalogue` (L2315),
`persist_planning_artifacts_v2` (L2344), `run_planner_v2` (L3080),
`build_repair_planner_prompt` (L2048),
`REPAIR_PLANNER_INLINE_TARGET_BYTES` (L1917), `REQUIRE_STAGED_POLICY_TEXT` (L1691).

- `git grep` for each name over `src` + `tests` + `docs` -> hits only inside
  `planning_v2.py`: the definition line, the `__all__` entry, and — for the
  three aliases — their right-hand side.
- Reachability from non-`__all__` roots: **unreachable**. Their live
  counterparts are reachable and must stay: `write_implementation_bundle`
  (called at L2341, L2378, L2860) and `persist_planning_v2_artifacts`
  (called by `PlannerV2` at L2591 and `RepairPlannerV2` at L2999, both the
  live writers of `task_plan.json`).
- Verdict: **UNCERTAIN** — all seven are in `__all__`, so they are a published
  API promise rather than proven-dead code. Deleting them is safe only after v3
  declares the public planning API.

### D7. `RunStatus.BLOCKED` (`src/metaharness/models.py:239`)

- `git grep -n 'RunStatus\.BLOCKED'` -> **2** hits, both comparisons in
  `orchestrator._diagnose_result` (L1558, L1585).
- Status-writer scan over `update`, `update_if_status`, `transition_if`,
  `initialize`, `_write_checkpoint`: **0** writes. The value is only read, and
  is a recognized state key for plan recovery
  (`plan_recovery.RECOVERABLE_PLANNER_STATE_PAIRS`, `("blocked", "PLANNER_BLOCKED")`).
- Verdict: **UNCERTAIN** — legacy run status. Invariant 2 requires old runs to
  stay readable, so the enum member and the recovery pair stay until the v3
  reader explicitly owns them.

### D8. `UIConfig.default_implementer_profile` (`src/metaharness/models.py:540`)

- `git grep` -> `config.py:532,540,990,1047`, `run_options.py:163,444`,
  `models.py:540`, and 8 test modules.
- Importers: none directly. The loader always passes `implementer_default =
  None` (`config.py:990-991`), so the dataclass field is only populated by
  hand-built configs in tests (`tests/test_web_security.py:46`).
- Entrypoint: none in production. The *config data key*
  `[ui] default_implementer_profile` is a different, still-supported input path
  (`config._routing`, L532-540) documented in `docs/providers.md:77`.
- Verdict: **UNCERTAIN** — the field is dead in production but the legacy key it
  mirrors is documented input. Removing the field alone would break
  `effective_run_config` (`run_options.py:440`) and 8 test modules.

### D9. `RunOptions.default_implementer_profile` (`src/metaharness/run_options.py:75`)

- `git grep` -> `run_options.py:73-86,163,444`, plus
  `tests/test_pipeline_v2_invariants.py:242` and
  `tests/test_generic_pipeline_v2.py:71` (both pass it explicitly).
- Importers: none.
- Verdict: **ACTIVE** (test-only consumer). It is a deliberate constructor-only
  migration aid, asserted absent from the serialized snapshot by
  `tests/test_frozen_routing.py:164` — deleting it is a v3 schema decision.

### D10. Legacy resume/recovery paths: `stranded_contract_repair`,
`_check_repair_exhaustion_info`, `historical_step_acceptance`,
`plan_recovery.RECOVERABLE_PLANNER_STATE_PAIRS`

- `resume.stranded_contract_repair` -> called by `resume.resume_info` and
  covered by `tests/test_contract_repair_output_correction.py:715`
  (`StrandedRunRecoveryTests`, including "reproduce the legacy stranded state
  without a legacy code path").
- `resume._check_repair_exhaustion_info` -> covered by
  `tests/test_pipeline_v2_machine.py:1109` and `:1163`, which assert
  `resume.migration == "historical_check_repair_redaction_crash"`.
- `step_authority.historical_step_acceptance` -> covered by
  `tests/test_step_authority.py:464` (`HistoricalStaleAuthorityTests`).
- `plan_recovery.RECOVERABLE_PLANNER_STATE_PAIRS` -> 7 state/failure pairs; the
  `("blocked", "PLANNER_BLOCKED")` pair and the three "once projected onto their
  waiting conditions" pairs are legacy shapes.
- Verdict: **ACTIVE**. All three paths have live consumers and tests; they are
  compatibility branches to be *relocated*, not deleted (invariant 2). The
  `plan_recovery` legacy pairs are the only ones with no test that asserts the
  legacy shape itself — **UNCERTAIN** for that subset alone.
