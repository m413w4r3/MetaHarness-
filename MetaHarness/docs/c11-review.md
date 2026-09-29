# Revue et corrections C11

Base : `06831937173723ba82e78e03af24a51d2e34ad8b`.

**C11 reste ouvert : le budget de lignes physiques n'est pas atteint.**
Les corrections sont livrées pour revue, sans annoncer la fermeture de C11.
Le garde de structure reste strict et échoue sur ce seul budget.

| Autorité runtime | BEFORE | AFTER | Cible |
| --- | ---: | ---: | ---: |
| Lignes Python physiques sous `src/metaharness` | 36 371 | 34 631 | ≤ 32 000 — échec |
| Modules directs `orchestration`, `__init__.py` inclus | 26 | 14 | ≤ 14 |
| RunPhase | 10 | 10 | ≤ 10 |
| RunStatus | 19 | 7 | ≤ 10 |
| Reasons canoniques détectés chez les producteurs, PARTIAL inclus | 80 | 39 | ≤ 40 |
| Champs AutonomyBudget | 5 | 5 | exactement 5 |

## P2 — réduction de taxonomie (2026-09-29)

| Mesure P2 | Avant | Après | Cible |
| --- | ---: | ---: | ---: |
| RunStatus | 9 | 7 | ≤ 10 |
| RunPhase | 10 | 10 | ≤ 10 |
| Reasons durables canoniques | 40 | 39 | ≤ 40 |
| Champs AutonomyBudget | 5 | 5 | exactement 5 |

RUNSTATUS BEFORE : `CREATED, RUNNING, WAITING_HUMAN, WAITING_EXTERNAL,
COMMITTED, PUBLISHED, PARTIAL, FAILED, INTERRUPTED`.
RUNSTATUS AFTER : `RUNNING, WAITING_HUMAN, WAITING_EXTERNAL, COMMITTED,
PUBLISHED, PARTIAL, FAILED`.

Les usages runtime avant la réduction étaient dans `models.py` (projection et
bridge), `state.py` (initialisation et persistance), `result.py` (résultat),
`resume.py` (table phase/statut), `orchestration/runtime.py` (trace et
diagnostics), `cli.py` (sortie 130), et `web/api.py`, `web/pages.py` et
`web/static/run.js` (contrôle, présentation et polling). Les tests couvraient
`test_state`, `test_step_authority`, `test_recovery_policy`, les suites
`pipeline`/`autonomy`, `test_web_api` et `test_web_pages`. `CREATED` dérivait
de `phase=None` sous RUNNING ; `INTERRUPTED` de FAILED + reason INTERRUPTED.
Les statuts `COMMITTED`, `PUBLISHED` et `PARTIAL` restent les résultats
terminaux publics distincts. PLAN_REJECTED reste reason sous FAILED.

Inventaire des tests qui lisaient chaque membre avant P2 : `CREATED` —
`test_state.py`; `RUNNING` — `test_state.py`; `WAITING_HUMAN` —
`test_c11_structure.py`, `test_state.py`, `test_recovery_policy.py`,
`pipeline/test_multi_iteration.py`, `pipeline/test_autonomy_contract.py`,
`pipeline/test_budget.py`, `pipeline/test_worker_recovery.py`,
`pipeline/test_per_step_gate.py`, `autonomy/test_resume_external.py`,
`autonomy/test_recovery_default.py`, `autonomy/support.py`;
`WAITING_EXTERNAL` — `test_state.py`, `test_plan_repository_validation.py`,
`pipeline/test_resume.py`, `pipeline/test_multi_iteration.py`,
`pipeline/test_budget.py`, `pipeline/test_worker_recovery.py`,
`pipeline/test_remote.py`, `pipeline/test_per_step_gate.py`,
`autonomy/test_resume_external.py`, `autonomy/test_transport.py`;
`COMMITTED` — `test_state.py`, `autonomy/test_resume_external.py`,
`autonomy/support.py`; `PUBLISHED` — `test_state.py`,
`test_step_authority.py`, `test_plan_repository_validation.py`,
`pipeline/test_lifecycle.py`, `pipeline/test_resume.py`,
`pipeline/test_multi_iteration.py`, `pipeline/test_autonomy_contract.py`,
`pipeline/test_budget.py`, `pipeline/test_worker_normalization.py`,
`pipeline/test_worker_recovery.py`, `pipeline/test_remote.py`,
`pipeline/test_per_step_gate.py`, `autonomy/test_resume_external.py`,
`autonomy/support.py`, `autonomy/test_transport.py`; `PARTIAL` —
`test_c11_structure.py`, `pipeline/test_multi_iteration.py`,
`pipeline/test_budget.py`, `autonomy/test_resume_external.py`,
`autonomy/support.py`; `FAILED` — `test_state.py`, `test_scope.py`,
`test_recovery_policy.py`, `test_step_authority.py`,
`test_plan_repository_validation.py`, `pipeline/test_resume.py`,
`pipeline/test_worker_recovery.py`, `pipeline/test_remote.py`,
`autonomy/test_resume_external.py`, `autonomy/test_recovery_default.py`,
`autonomy/support.py`; `INTERRUPTED` — `test_step_authority.py`.

Reasons durables BEFORE (40) : `AGENT_AUTH_FAILURE`,
`AGENT_CONTRACT_MISMATCH`, `AGENT_GIT_VIOLATION`, `AGENT_RUNTIME_FAILED`,
`AGENT_SCOPE_VIOLATION`, `AGENT_TIMEOUT`, `AUDIT_PROFILE_NOT_WRITABLE`,
`BASE_MOVED_SINCE_RUN`, `CHECK_FAILED`, `CHECK_INFRASTRUCTURE_UNAVAILABLE`,
`CHECK_SIDE_EFFECT_REPEATED`, `COMMIT_GATE_FAILED`, `COMMIT_SECURITY_FAILURE`,
`CONFIGURATION_INVALID`, `DETERMINISTIC_GATE_FAILED`,
`DURABLE_ARTIFACT_CORRUPTED`, `EXTERNAL_AUTH_REQUIRED`,
`GITHUB_WORKSTREAM_FAILURE`, `GIT_FAILURE`, `HARD_DENY_PATH_MUTATION`,
`INTERNAL_HARNESS_ERROR`, `LLM_TRANSPORT_EXHAUSTED`, `PAUSED`,
`PER_STEP_GATE_REGRESSION`, `PLANNER_OUTPUT_INVALID`, `PLAN_APPROVAL_INVALID`,
`PLAN_REJECTED`, `PUSH_FAILED`, `REPOSITORY_TREE_DRIFT_UNEXPLAINED`,
`RESUME_INTEGRITY_FAILURE`, `RESUME_REQUIRES_OPERATOR`, `ROLLBACK_FAILED`,
`RUN_SCHEMA_UNSUPPORTED`, `SPEC_DECISION_REQUIRED`,
`TREE_MODIFIED_OUTSIDE_AUTHORITY`, `cost_cap`, `max_iterations`,
`stagnation`, `wall_clock`.

Reasons durables AFTER : même vocabulaire sans `AGENT_AUTH_FAILURE`; cet alias
et les entrées `LLM_401`, `LLM_403`, `LLM_AUTH_FAILURE` se normalisent vers
`EXTERNAL_AUTH_REQUIRED` avant persistance. Les anciennes valeurs de RunStatus
restent uniquement dans le bridge de lecture `disposition_for_status`; aucune
n’est un membre runtime. Le bridge accepte `created`, `planning`,
`awaiting_plan_approval`, `preparing`, `implementing`, `validating`, `revising`,
`approved`, `publishing`, `paused`, `waiting_remote`, `plan_rejected` et
`interrupted`.

La projection RUNNING, WAIT_EXTERNAL, WAIT_HUMAN, PARTIAL, COMMITTED, PUBLISHED
et FAILED est testée. L’interruption reste le reason durable INTERRUPTED et
conserve le code de sortie CLI 130. Aucun changement de phase ni de machine
d’état n’a été ajouté.

Validation P2 : compileall, `git diff --check`, tests ciblés état, recovery,
CLI, résultat, reprise, Web/API, diagnostics et architecture ont été lancés.
La projection API sans serveur a réussi. Les suites démarrant un socket ne
passent pas la sandbox. Le garde historique de lignes source reste en échec ;
34 590 lignes existaient sur HEAD avant P2, 34 631 après (cible 32 000).
Quelques tests d’intégration de reprise injectant `RuntimeError` restent en
échec dans la projection préexistante `INTERNAL_HARNESS_ERROR` vers
`MARK_FAILED_CONTINUE`.

Le travail reçu comptait 36 302 lignes physiques. Son résultat de 31 897
excluait les lignes vides : ce compteur a été remplacé par
`len(read_text().splitlines())`. À la fin de la première passe, il restait
2 582 lignes à supprimer pour fermer C11. Le compteur des codes inspecte les producteurs par AST, et non les
motifs de policy. Il inclut les quatre raisons PARTIAL et les raisons
opérateur. Les expressions dynamiques d'extensions futures ne constituent
pas un inventaire fermé ; unknown conserve la policy FIXABLE de v4.

RunPhase conservées : `CONTEXT, PLANNER, PLAN_APPROVAL, WORKTREE_SETUP,
IMPLEMENT_STEP, DETERMINISTIC_GATE, AUDIT, CANDIDATE_READY, CANDIDATE_PUSH, PUBLISH`.

Taxonomie RunStatus historique v4 : `CREATED, PLANNING, WAITING_HUMAN, AWAITING_PLAN_APPROVAL,
WAITING_EXTERNAL, PAUSED, WAITING_REMOTE, PLAN_REJECTED, PREPARING,
IMPLEMENTING, VALIDATING, REVISING, APPROVED, PUBLISHING, PUBLISHED,
COMMITTED, PARTIAL, FAILED, INTERRUPTED`.
RunStatus après la refondation initiale : `CREATED, RUNNING, WAITING_HUMAN,
WAITING_EXTERNAL, COMMITTED, PUBLISHED, PARTIAL, FAILED, INTERRUPTED`.
Les états v4 déjà écrits sont projetés depuis leur disposition et leur phase,
sans modifier leurs fichiers lors d'une lecture HTTP.

AutonomyBudget : `step_attempts, audit_repairs, max_iterations,
max_wall_clock_hours, max_cost`, avant et après.
Aucun module d'orchestration sans importeur runtime n'a été identifié : les
suppression de fichiers d'orchestration sont des fusions de responsabilités.

## DELETED

- Parseur `llm/wire.py`, protocole ScopeRequest et transport de pièces jointes
  sans consommateur runtime ; tests de ces surfaces retirées.
- `repository_topology.py`, sans consommateur, et rendu de diff sémantique
  devenu inutilisé ; helpers morts de diagnostic, acceptation et restauration.
- Dataclass WorkstreamRef sans consommateur et RecoveryStepUnavailable sans
  action associée ; imports et branches devenus redondants après les fusions.
- Branches des schémas d'approbation 2 à 4, impossibles pour les checkpoints
  v4 actuels ; le schéma 1 reste l'API standalone, le schéma 5 lie v4.
- Motifs FATAL ajoutés par le travail reçu qui capturaient des codes inconnus.

## MERGED

- `audit_prompt`, `audit_protocol` → `audit`.
- `check_failure`, `gate_acceptance`, `per_step_gate` → `gates`.
- `candidate` → `publication`.
- `check_recovery`, `worker_recovery` → `recovery`.
- `durable_readers` → `shared` ; `run_observability` → `runtime`.
- `step_authority`, `step_acceptance` → `step_execution`.
- Variantes d'erreurs chez leurs producteurs et consommateurs : processus,
  infrastructure, configuration, validation de commit et frontières FATAL.
  Les secrets, blobs non inspectables et tailles de diff gardent leur détail
  après le code stable. Les raisons PARTIAL restent distinctes.

## KEPT INTENTIONALLY

- Les classifications exactes des anciennes frontières FATAL enregistrées
  par v4 restent en lecture ; elles ne sont plus produites. Aucun nouveau
  motif FATAL ne capture un namespace inconnu.
- Le détail backend `rate_limited` reste distinct d'une panne générique :
  seul ce détail admet le fallback d'audit, sans accepter un rapport issu
  d'une exécution échouée. Le code durable commun est AGENT_RUNTIME_FAILED.
- Approbation générique optionnelle ; autorité des checks gelée ; Git-first
  resume ; auto-resume et WAIT_EXTERNAL ; AUDIT writable ; PlannerContinue
  multi-milestone ; PARTIAL terminal autonome ; FAILED_CONTINUED ; budget C10.
- `max_steps_per_plan = 21`, approval AutoWork false ; seuils 6/12/18/8500 ;
  per-step test-collection ; audit_max_bytes 64000 ; fallback codex-sol-medium ;
  modèle Claude claude-opus-5-5 ; diff accessible par artefact et prompts bornés.
- L'UI suit la phase malgré un statut RUNNING constant. L'approbation arrête
  le polling et garde sa CSP. Les diagnostics retrouvent les contrats de
  l'itération et conservent les résumés de steps.

## Checks exécutés

- Une campagne complète dans cette passe :
  `.venv/bin/python scripts/test_parallel.py -j 4`, 68 modules, 81,61 s.
  Résultat initial : 882 tests comptabilisés, une erreur d'import Git et une
  erreur fonctionnelle d'audit, plus le dépassement du budget de lignes.
- Les deux erreurs ont été corrigées et les modules concernés repris :
  Git/validation 39 tests OK ; audit/Claude/contrats d'exécution/worker recovery
  41 tests OK. Tests ciblés de reprise/UI/architecture/policy : 97 tests OK.
- La découverte finale compte **905 tests sur 68 modules**. Après reprises,
  **904 passent ; seul le garde source ≤ 32 000 reste en échec**. Aucune
  défaillance fonctionnelle restante observée ; ce n'est pas une suite verte.
- Dernière reprise approval/policy/budgets : 32 tests, un échec LOC uniquement.
- Compileall src/tests, Ruff ciblé F/E9, Node `--check` du script UI et
  `git diff --check` : OK. Mypy, pyright et ty absents du PATH : pas de
  typecheck prétendu exécuté. Aucun appel fournisseur réel dans ces tests.

MAX STEPS PER PLAN = 21 ; AUTOWORK PLAN APPROVAL = FALSE.
RUNPHASE ≤ 10 ; RUNSTATUS ≤ 10 ; FAILURE CODES ≤ 40.
AUTONOMY BUDGET HAS EXACTLY 5 FIELDS ; ORCHESTRATION ≤ 14 MODULES.
WAIT_HUMAN ONLY FOR SPEC_DECISION ; NO LEGACY REPAIR PIPELINE REINTRODUCED.
NO NEW AUTONOMY MECHANISM ADDED.
**SRC ≤ 32000 LINES : NON. C11 STRUCTURAL BUDGETS CLOSED : NON.**

## P3 deletion-first orchestration pass (2026-09-29)

### Metrics

| METRIC | BEFORE (HEAD) | AFTER | LIMIT |
| --- | ---: | ---: | ---: |
| src/metaharness physical Python lines | 34,631 | 34,545 | ≤ 32,000 |
| Direct orchestration modules, excluding __init__.py | 13 | 13 | ≤ 14 |
| RunPhase | 10 | 10 | ≤ 10 |
| RunStatus | 7 | 7 | ≤ 10 |
| Durable failure/reason codes | 39 | 39 | ≤ 40 |
| AutonomyBudget fields | 5 | 5 | exactly 5 |

The source budget remains open by 2,545 lines. The focused C11 test measured
the current tree and fails only test_source_line_budget. The full-suite run
preceded a final comment/self-import cleanup and measured 34,549 lines.

### Orchestration survey

There were no direct orchestration modules at or below 100 lines. The single
runtime-caller modules were audit, run_composition, run_failure,
step_execution, and worker_attempt; each owns a distinct service, so none
is a wrapper to delete. Caller counts were unchanged by this pass:

| Module | Callers |
| --- | ---: |
| audit | 1 |
| gates | 4 |
| pipeline_v2 | 11 |
| publication | 4 |
| recovery | 4 |
| run_bootstrap | 2 |
| run_composition | 1 |
| run_failure | 1 |
| run_resume | 2 |
| runtime | 9 |
| shared | 12 |
| step_execution | 1 |
| worker_attempt | 1 |

The internal dependency graph is unchanged except for removal of self-imports
from gates and runtime: audit → pipeline_v2, publication, runtime, shared;
gates → pipeline_v2, runtime, shared; pipeline_v2 → —;
publication → pipeline_v2, recovery, runtime, shared;
recovery → pipeline_v2, shared; run_bootstrap → gates, pipeline_v2, runtime,
shared; run_composition → gates, pipeline_v2, publication, run_bootstrap,
runtime, shared; run_failure → pipeline_v2, recovery, run_resume, runtime,
shared; run_resume → shared; runtime → audit, gates, pipeline_v2, publication,
recovery, run_bootstrap, run_composition, run_failure, shared, step_execution,
worker_attempt; shared → pipeline_v2; step_execution → gates, pipeline_v2,
publication, recovery, runtime, shared; worker_attempt → runtime, shared.

Production orchestration private imports from siblings fell from 13 to 0.
Two test-only private probes remain (audit._evidence_payload and
cli._auto_resume) for direct protocol assertions.

### Deleted, merged, and retained

- Deleted modules: none in this pass. There were no remaining wrapper modules
  under the direct orchestration package.
- Merged modules: no additional file merges were justified. HEAD already has
  audit_prompt/audit_protocol → audit;
  check_failure/gate_acceptance/per_step_gate → gates;
  candidate → publication; check_recovery/worker_recovery → recovery;
  run_observability → runtime; and step_authority/step_acceptance →
  step_execution.
- Deleted helpers: unused read_tree_file, create_file_once,
  _PLANNER_ATTEMPT_ARTIFACTS, the unused agent-report limit alias, and
  redundant private aliases/self-imports.
- Moved helper: check result projection and its attempt artifact list now live
  in gates; publication imports the worktree status predicate directly from
  attempt_transaction.
- Shared helpers: sibling consumers use public spellings for JSON, attempt
  archival, tree, and Git identity helpers. shared retains helpers with
  multiple orchestration consumers.
- Kept modules: audit (writable bounded protocol and repairs), gates (frozen
  check authority and regression-only blocking), pipeline_v2 (milestone
  lifecycle), publication (candidate and remote boundary), recovery
  (classification and bounded retries), run_bootstrap (initial setup),
  run_composition (continuation wiring), run_failure (settlement), run_resume
  (Git-first checkpoint recovery), runtime (composition and observability),
  shared (multi-consumer primitives), step_execution (scope, acceptance and
  commit), and worker_attempt (one executor attempt).

The required legacy-path search found no runtime contract/check repair or
reviser pipeline, old plan-recovery service, RecoveryBudgets, or retired retry
knobs. STEP_ACCEPTANCE remains only as an artifact name, not a phase. Current
resume integrity checks and the guard rejecting a root-level legacy
execution-selection artifact remain active authority checks.

### Checks and final confirmations

- Targeted runner: 206 tests across 15 modules; 10 modules passed and 5
  exposed the already documented project_exit/MARK_FAILED_CONTINUE
  exception-projection defect. test_config, architecture boundaries,
  recovery policy, lifecycle/publication, audit, gate baseline, planner
  normalization, generic pipeline, and worker recovery passed.
- Final structural/architecture selection: 36 tests; 35 passed, with only the
  source LOC budget failing.
- Full suite, .venv/bin/python scripts/test_parallel.py -j 4: 913 tests in
  70 modules; 1 failure, 155 errors, 3 skipped; 71.10 seconds. Forty errors
  were the known exception-projection defect; 115 were local-socket
  PermissionError failures under the sandbox.
- compileall, import smoke check, and git diff --check: passed.

SRC <= 32000 LINES: NO (34,545). ORCHESTRATION <= 14 MODULES: YES (13).
RUNPHASE <= 10: YES (10). RUNSTATUS <= 10: YES (7).
DURABLE FAILURE CODES <= 40: YES (39).
AUTONOMY BUDGET HAS EXACTLY 5 FIELDS: YES.
AUTOWORK MAX_STEPS_PER_PLAN = 21: YES.
AUTOWORK REQUIRE_PLAN_APPROVAL = FALSE: YES.
WAIT_HUMAN ONLY FOR SPEC_DECISION: YES. FATAL ONLY HARD-STOPS: YES.
FIXABLE EXHAUSTION CONTINUES AUTONOMOUSLY: YES.
TRANSIENT EXHAUSTION WAITS EXTERNALLY AND IS RESUMABLE: YES.
NO LEGACY REPAIR PIPELINE REINTRODUCED: YES.
NO NEW AUTONOMY MECHANISM ADDED: YES.
C11 STRUCTURAL BUDGETS CLOSED: NO; the physical source-line limit remains
the sole unmet structural budget.
