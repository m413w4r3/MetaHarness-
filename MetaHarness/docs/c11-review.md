# Revue et corrections C11

Base : `06831937173723ba82e78e03af24a51d2e34ad8b`.

**C11 reste ouvert : le budget de lignes physiques n'est pas atteint.**
Les corrections sont livrées pour revue, sans annoncer la fermeture de C11.
Le garde de structure reste strict et échoue sur ce seul budget.

| Autorité runtime | BEFORE | AFTER | Cible |
| --- | ---: | ---: | ---: |
| Lignes Python physiques sous `src/metaharness` | 36 371 | 34 582 | ≤ 32 000 — échec |
| Modules directs `orchestration`, `__init__.py` inclus | 26 | 14 | ≤ 14 |
| RunPhase | 10 | 10 | ≤ 10 |
| RunStatus | 19 | 9 | ≤ 10 |
| Codes détectés chez les producteurs, raisons PARTIAL incluses | 80 | 40 | ≤ 40 |
| Champs AutonomyBudget | 5 | 5 | exactement 5 |

Le travail reçu comptait 36 302 lignes physiques. Son résultat de 31 897
excluait les lignes vides : ce compteur a été remplacé par
`len(read_text().splitlines())`. Il reste 2 582 lignes à supprimer pour fermer
C11. Le compteur des codes inspecte les producteurs par AST, et non les
motifs de policy. Il inclut les quatre raisons PARTIAL et les raisons
opérateur. Les expressions dynamiques d'extensions futures ne constituent
pas un inventaire fermé ; unknown conserve la policy FIXABLE de v4.

RunPhase conservées : `CONTEXT, PLANNER, PLAN_APPROVAL, WORKTREE_SETUP,
IMPLEMENT_STEP, DETERMINISTIC_GATE, AUDIT, CANDIDATE_READY, CANDIDATE_PUSH, PUBLISH`.

RunStatus initial : `CREATED, PLANNING, WAITING_HUMAN, AWAITING_PLAN_APPROVAL,
WAITING_EXTERNAL, PAUSED, WAITING_REMOTE, PLAN_REJECTED, PREPARING,
IMPLEMENTING, VALIDATING, REVISING, APPROVED, PUBLISHING, PUBLISHED,
COMMITTED, PARTIAL, FAILED, INTERRUPTED`.
RunStatus final : `CREATED, RUNNING, WAITING_HUMAN, WAITING_EXTERNAL,
COMMITTED, PUBLISHED, PARTIAL, FAILED, INTERRUPTED`.
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
