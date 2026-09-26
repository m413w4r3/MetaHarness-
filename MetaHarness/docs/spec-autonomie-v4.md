# SPEC — MetaHarness v4 : l'autonomie d'abord

Statut : proposition. Base : `5656d4c` (PROMPT 23). Remplace les invariants 4-6
de `docs/refoundation-v3.md` ; conserve 1, 2, 3, 7, 12.

---

## 0. État constaté après les 23 commits

### 0.1 Ce qui a marché
- `orchestrator.py` : 7788 → 488 lignes. C'est maintenant une vraie façade.
- `planning_v2.py`, `check_repair.py`, `resume_validation.py` : découpés.
- Une seule machine d'état (`RunMachineState` = phase + disposition + reason).
- Compatibilité historique largement supprimée (RunOptions, resume, alias).

### 0.2 Ce qui n'a pas marché

| Constat | Mesure |
| --- | --- |
| La simplification globale n'a pas eu lieu | `src/` 43 513 → **48 527** lignes (+11 %), `tests/` 21 728 → 25 513 |
| La logique est redistribuée, pas réduite | `orchestration/` = 36 modules ; recovery ≈ 7 800 lignes ; resume/intégrité ≈ 4 800 lignes |
| HEAD est rouge | 7 tests en échec : `test_orchestrator_e2e::test_commit_is_impossible_without_both_gates` (`waiting_contract_repair` au lieu de `waiting_check_repair`), `test_prompt_contracts` (check_repair.txt), 5× `test_web_api` (`RunStateStore.update` n'existe plus) |
| Aucun run réel depuis la refonte | Derniers runs : 2026-09-23. Les 23 commits n'ont été validés que par des tests unitaires |
| **0 run réel sur 5 a abouti** | voir 0.3 |
| La refonte a été écrite par l'exécutant low-tier sans revue | Les 24 commits portent `Modèle: deepseek-flash`. Les prompts 3, 4, 11, 12 et 13 ont *ajouté* des mécanismes (+1,1k à +1,6k lignes chacun) au lieu de relâcher des contraintes |

### 0.3 Pourquoi les runs réels meurent

| Run | Cause | Cause racine dans le code |
| --- | --- | --- |
| 3 runs | `LLM_FAILURE: HTTP 503 after 3 attempt(s)` → `failed` | `llm/chat.py:166-167` : backoff de 0,05 s à 1 s max. Trois essais en moins d'une seconde contre un bridge surchargé. L'échec du planner n'est pas un `WAIT_EXTERNAL` reprenable : c'est un `failed` |
| 1 run | `STEP_CONTRACT_DRIFT create_exists=…/reference_corpus.py` → `failed` | Le planner a mis en CREATE un fichier qui existe. `step_authority.py:192` le détecte, et le code n'est pas classé : `UNKNOWN` → `HARD_STOP`. Une normalisation CREATE→WRITE aurait suffi |
| 1 run | `CHECK_REPAIR_EXHAUSTED` sur `test-integration` → `waiting_check_repair` | (a) pas de baseline : 20 tests d'intégration en échec, dont une partie sans rapport avec la tâche (`test_source_access_policy`) et probablement déjà en échec sur `main` ; (b) le worker de réparation (`codex-luna-high`, sandbox) n'a pas accès à Docker, donc le prompt l'oblige à répondre `BLOCKED/NOT_RUN` ; (c) budget de 2 ; (d) une réparation transverse est confiée au tier le plus faible |

### 0.4 La politique de recovery est fail-closed

`recovery_policy.py` : un code inconnu donne `UNKNOWN` → `HARD_STOP`. En
classant tous les codes levés par littéral dans `src/` :

- **11 codes levés par le pipeline ne sont pas classés.** S'ils sortent de leur
  boucle locale, ils deviennent un `HARD_STOP` : `REPLAN_EXECUTION_SELECTION_MISMATCH`,
  `REPLAN_CYCLE_REQUIRED`, `REPAIR_SCOPE_EXPANSION`, `REPAIR_PLANNER_BLOCKED`,
  `REVIEW_ROUTE_NOT_NONE`, `REPAIR_SCOPE_UNJUSTIFIED`,
  `REPAIR_SCOPE_EXISTING_PATH_MISSING`, `REPAIR_SCOPE_CREATE_PATH_EXISTS`,
  `INVALID_PHASE_TRANSITION`, `IMPLEMENTATION_CORRECTION_CYCLE_REQUIRED`,
  `STEP_CONTRACT_DRIFT`. `run_failure._failure_reason` produit aussi
  `GIT_FAILURE`, `TOCTOU_FAILURE`, `INTERNAL_HARNESS_ERROR`,
  `EXECUTION_SELECTION_INVALID` et `CHECK_SETUP_INVALID`, tous en `HARD_STOP`.
- Un worker low-tier qui touche **un seul** fichier hors `WRITE_SET` (un test
  voisin, un `__init__.py`) déclenche `AGENT_SCOPE_VIOLATION`, classé
  `AUTHORITY` → `HARD_STOP` (`worker_attempt.py:482-488`,
  `worker_recovery.py:109-116`). Il n'y a ni rollback partiel ni retry. C'est
  le mode d'échec le plus probable d'un exécutant low-cost.
- `AGENT_GIT_VIOLATION` (le worker a fait un commit) → `HARD_STOP`, alors
  qu'un `reset --soft` rattrape la situation.
- Chaque échelle se termine par `WAIT_HUMAN`. Aucune ne se termine par
  « escalader au tier supérieur » ni par « publier le dernier état vert ».

### 0.5 Le flux ne correspond pas à l'intention

| Intention | Réalité |
| --- | --- |
| Beaucoup de prompts simples et très détaillés pour des exécutants low-cost | `planner_v2.txt` : « smallest number of coherent steps », « Do not split a change merely to make steps smaller », contrat de 1000-2200 caractères, au plus 6 instructions, « do not explain » |
| Un reviewer high-tier qui bouche les trous, refactore et corrige | Le reviewer est **read-only**, passe par le bridge chat avec un extrait de diff borné, et le prompt dit « Do not request optional cleanup ». Le reviser (Claude) est borné au `APPROVED MUTABLE SCOPE` et ne corrige que le code mort *introduit*. `revision.enabled` vaut `False` par défaut (`config.py:807`) |
| Retour au planner pour audit et suite, en boucle | `PASS` → `publish` → fin. Aucune notion de jalons ni de lot suivant. Une SPEC qui dépasse `max_steps_per_plan` est impossible. `max_correction_cycles` vaut 1 par défaut |
| La sécurité ne doit pas entraver le pipeline | Environ 60 codes mènent à `HARD_STOP`, avec 3 portes humaines (approbation du plan, approbation du scope, épuisement des réparations) |

---

## 1. Principes v4 (nouveaux invariants)

- **A1. Progresser par défaut.** Seule une allowlist fermée et courte arrête un
  run (§C1). Tout le reste, codes inconnus compris, déclenche une action
  autonome.
- **A2. Normaliser plutôt que refuser.** Si le harness peut corriger une
  sortie de modèle de façon déterministe (CREATE sur un fichier existant,
  chemin READ absent, commit parasite, fichier hors scope inoffensif), il la
  corrige, la journalise et continue.
- **A3. Escalader plutôt qu'attendre.** Échelle d'escalade : même exécutant
  avec le feedback, puis tier supérieur, puis replanification du step par le
  planner. Ensuite on marque l'échec et on continue avec le dernier état vert.
  `WAIT_HUMAN` n'existe que pour une vraie décision produit (`SPEC_DECISION`).
- **A4. Le harness exécute les checks.** Un worker n'est jamais tenu de
  prouver qu'il a lancé un check. Le gate du harness est la seule preuve.
- **A5. Seules les régressions comptent.** Un check déjà rouge sur la base ne
  bloque jamais un run.
- **A6. Git est la source de vérité.** Chaque étape réussie devient un commit.
  Une reprise repart du dernier commit et rejoue l'étape en cours ; elle ne
  prouve pas l'intégrité d'une chaîne d'artefacts.
- **A7. Supprimer vaut mieux qu'ajouter.** Chaque chantier indique les
  modules qu'il supprime. Aucun chantier ne doit augmenter `src/` en net, sauf
  C8.
- **A8. Le travail architectural va au tier high.** Les prompts qui
  restructurent (C7, C8, C9) sont implémentés ou revus par un modèle
  high-tier. Seuls les prompts mécaniques vont aux exécutants low-cost.

---

## 2. Chantiers

Chaque chantier donne le problème, la solution, ce qu'il supprime et ses
critères d'acceptation. L'ordre d'exécution est au §3.

### C0 — Remettre HEAD au vert
- Corriger les 7 tests (0.2). Pour `test_commit_is_impossible_without_both_gates`,
  décider du statut attendu *après* C1 plutôt que de figer l'ancien.
- Supprimer `tests/test_p15.py`, `test_p17.py`, `test_p19.py`, `test_p23.py`,
  `test_p27.py`, `test_prompt4_commit_chain.py` s'ils ne testent que
  l'historique d'une campagne. Sinon, les renommer par comportement testé.
- **Acceptation** : suite complète verte, lancée avec `-n 8` en moins de 7 min.

### C1 — Inverser la politique d'arrêt (`recovery_policy.py`)

**Solution**
1. Passer de 9 `FailureClass` à 4 :
   - `TRANSIENT` : transport, infrastructure, timeouts.
   - `FIXABLE` : correctness, contrat, protocole modèle, **inconnu**.
   - `SPEC_DECISION`.
   - `FATAL`.
2. `FATAL` est une allowlist exhaustive, et le seul chemin vers `HARD_STOP` :
   - un secret détecté dans le diff ou un blob stagé (`SECRET_*`,
     `*_BLOB_NOT_REVIEWABLE`) ;
   - une écriture hors du worktree ou dans `.git/` ;
   - une ref autre que la branche du run modifiée, ou un push effectué par un
     agent ;
   - une modification d'un chemin de la **hard-deny list** (§C2) ;
   - un rollback impossible (`ROLLBACK_FAILED`), c'est-à-dire un worktree
     irrécupérable ;
   - l'épuisement du budget global (§C10).
3. Toutes les échelles `FIXABLE` suivent : `RETRY_WITH_FEEDBACK` →
   `ESCALATE_TIER` → `REPLAN_STEP` → `MARK_FAILED_CONTINUE`. La dernière
   entrée fait un rollback du step, enregistre l'échec dans le rapport
   d'itération et passe à l'audit (C7). L'audit décide s'il faut compléter.
4. `TRANSIENT` : backoff long (C4) → fallback d'exécutant → `WAIT_EXTERNAL`
   avec reprise automatique.
5. Un code non classé produit l'événement `recovery.unclassified_code` et
   l'échelle `FIXABLE`. Ajouter un test qui énumère les codes levés dans `src/`
   et vérifie que chacun est classé explicitement : le garde-fou porte sur la
   *classification*, pas sur l'arrêt.

**Supprime** : `_HARD_STOP_CODES`, `_OPERATOR_DECISION_CODES`,
`terminal_strategy`, `_CORRECTNESS_REPAIR_CODES`, les tables dupliquées
`*_STRATEGY_CODES`. Cible : `recovery_policy.py` ≤ 300 lignes (908
aujourd'hui).

**Acceptation**
- `classify_failure("WHATEVER_NEW")` renvoie une stratégie non terminale.
- Seuls les codes de l'allowlist `FATAL` renvoient `HARD_STOP`.
- Aucune échelle ne se termine par `WAIT_HUMAN`.

### C2 — Le scope devient un signal, plus une barrière

**Solution**
1. Définir une **hard-deny list** dans la config, avec ces valeurs par
   défaut : `.git/**`, `.github/workflows/**`, `.env*`, `**/*secret*`, les
   fichiers de config MetaHarness et les checks du catalogue (Makefile cibles ?
   à décider, voir §4). Toute mutation de ces chemins est `FATAL`.
2. Le reste suit le mode `scope_mode = "soft"`, par défaut :
   - un chemin hors `WRITE/CREATE/DELETE_SET` et hors deny-list est **admis**,
     enregistré dans `step.json.out_of_scope_paths` et signalé à l'audit (C7) ;
   - avec `scope_mode = "strict"`, le harness restaure ces chemins depuis
     `tree_before` au lieu de s'arrêter. Il garde le reste du diff et relance
     le step avec le feedback si le diff restant est vide.
3. `AGENT_GIT_VIOLATION` est traité ainsi :
   - si le worker a commité sur la branche du run, le harness fait
     `git reset --soft <tree_before commit>` et poursuit sur le diff ;
   - s'il a créé une branche, le harness la supprime ;
   - s'il a touché une autre ref ou fait un push, c'est `FATAL`.
4. L'approbation de scope est supprimée : plus de `WAITING_SCOPE_APPROVAL`,
   `ScopeApprovalRequired` ni `scope_approval.json`, et les codes
   `REPAIR_SCOPE_*` disparaissent. Une demande de scope (`META SCOPE REQUEST`)
   est admise automatiquement hors deny-list.
5. Check-repair : les fichiers de test en échec sont éditables.

**Supprime** : la majeure partie de `orchestration/correction_scope.py` et de
`orchestration/check_scope.py`, les fonctions de scope de `approval.py`, et
`RepairScopePolicy`.

**Acceptation** : un faux agent qui modifie `WRITE_SET` plus un test voisin
produit un step `COMPLETED` avec `out_of_scope_paths=[test]`. Un faux agent qui
commite produit un step `COMPLETED`.

### C3 — Normalisation déterministe des contrats du planner

**Solution** : une fonction pure `normalize_step_contract(step, tree)`,
appliquée à la validation du plan **et** au début de chaque step :

| Contrat reçu | Correction appliquée |
| --- | --- |
| CREATE sur un chemin existant | passe en WRITE (et en READ) |
| WRITE ou DELETE sur un chemin absent | WRITE passe en CREATE ; un DELETE absent est retiré |
| READ absent | retiré du READ_SET, avec une note dans le contrat |
| Chemin présent dans deux sets | la priorité est WRITE > CREATE > DELETE |
| `REQUIRED_CHECKS` inconnu | retiré, avec les défauts ajoutés |
| `STEP_COUNT` incohérent | recalculé |

Chaque normalisation est écrite dans `plan.normalizations.json` et affichée
dans le rapport. Seule une contradiction réelle (un step dépend d'un fichier
qu'aucun step ne crée) part en correction du planner.

**Supprime** : `STEP_CONTRACT_DRIFT` en tant que failure, et la majeure partie
de `plan_repository_validation.py` (les préconditions deviennent des
normalisations).

**Acceptation** : rejouer le plan du run `20260923T131351Z-e9a34fd827` passe
S01 sans erreur.

### C4 — Transport LLM et agents résilients

**Solution**
1. `llm/chat.py` : backoff exponentiel avec jitter, de 2 s à 120 s par
   tentative. Respecter `Retry-After`. Horizon configurable
   (`transport.max_wait_seconds`, 1800 par défaut) pour 429, 5xx, timeouts et
   erreurs réseau.
2. À l'horizon atteint : `WAIT_EXTERNAL`, jamais `failed`, **y compris pour
   la phase PLANNER**, qui doit pouvoir être reprise.
3. Ajouter `metaharness run … --auto-resume` (et une option du serveur web) :
   une boucle qui relance les runs `WAIT_EXTERNAL` toutes les N minutes
   (10 par défaut), avec un plafond global.
4. Un timeout d'agent donne un retry avec le même contexte, puis le fallback
   d'exécutant.

**Acceptation** : un faux serveur qui renvoie 503 pendant 90 s puis 200 mène
le run à son terme sans intervention.

### C5 — Gate déterministe : baseline et checks exécutés par le harness

**Solution**
1. **Baseline** : avant le premier step, lancer les checks requis sur
   `base_sha`. Mettre le résultat en cache par
   `(base_sha, check_config_sha)` sous `runs_root/.baseline/`. Parser les IDs
   de tests en échec pour pytest, vitest et jest (sortie JUnit XML de
   préférence ; sinon regex `FAILED <id>`).
2. Verdict du gate :
   - **vert** : aucun nouvel ID en échec, et aucun check vert à la baseline
     devenu rouge ;
   - un check rouge à la baseline dont les IDs ne sont pas parsables devient
     `baseline_red` : il n'est pas bloquant et produit un avertissement.
3. **Infrastructure** : `preflight_argv` est évalué une fois par run. Si la
   preflight échoue, le check est `SKIPPED_INFRA` : un avertissement non
   bloquant, sauf si le check porte `blocking = true` dans le catalogue.
4. **Gate rapide par step** (optionnel, `gate.per_step = ["lint",
   "typecheck"]`) : juste après chaque step, le harness lance les checks
   rapides. En cas d'échec, `RETRY_WITH_FEEDBACK` se fait sur le **même**
   exécutant, qui a encore le contexte, avec l'extrait d'erreur. Une erreur
   d'un low-tier est bien plus facile à corriger au step N qu'au gate final.
5. Prompt check-repair : supprimer l'obligation « if no targeted check ran →
   BLOCKED/NOT_RUN ». Le worker édite et le harness relance le gate.
6. La réparation du gate final est confiée au **tier high** (voir C7),
   puisqu'elle est transverse par nature.

**Supprime** : `WAITING_CHECK_INFRASTRUCTURE` et la sémantique
`TARGETED_CHECK` du protocole de réparation.

**Acceptation** : un dépôt avec un test rouge sur `main` et une tâche qui ne le
touche pas donne un run vert, avec l'avertissement « baseline failure ».

### C6 — Planner : décomposition fine et contrats riches pour exécutants low-cost

**Solution** : réécrire `planner_v2.txt` (et le parser) selon ces règles.
1. **Objectif explicite** : « maximiser la probabilité qu'un modèle low-cost
   réussisse chaque step du premier coup ».
2. **Granularité** : 1 step correspond à une unité testable, avec 1 à 3
   chemins mutables par défaut. On découpe dès qu'un step exige deux
   raisonnements indépendants.
3. **Contrat** : 2 500 à 7 000 caractères, `max_step_contract_chars` = 9000
   par défaut. Sections du contrat :
   - `CONTEXT` : les faits du dépôt dont l'exécutant a besoin (signatures
     exactes, conventions, emplacement des tests) ;
   - `INSTRUCTIONS` : jusqu'à 12 opérations ;
   - `INTERFACES` : signatures ou schémas exacts à produire ;
   - `EXAMPLES` : un extrait de code si le motif n'est pas évident ;
   - `TESTS` : noms des tests à écrire et cas limites ;
   - `PITFALLS` : ce qu'un modèle faible ratera ;
   - `DONE_WHEN` ;
   - `VERIFY` : commandes exactes.
4. **Jalons** : si la SPEC dépasse un lot (`max_steps_per_plan`, 12 par
   défaut), le planner produit une section `MILESTONES` (M1…Mn, chacun avec
   ses critères d'acceptation) et planifie **seulement le prochain jalon**.
   C'est ce qui alimente C8.
5. `EXECUTION_CLASS` est conservé (routage vers un profil), avec une règle
   simple : MECHANICAL et REASONING vont au low-tier, AGENTIC au low-tier
   `max`, et l'escalade (A3) va vers le profil `escalation` de la classe.
6. Un `BLOCKED` du planner garde seulement `SPEC_DECISION`.
   `REPOSITORY_EVIDENCE` reste résolu automatiquement. `ATOMIC_SCOPE` est
   supprimé : un travail trop gros est découpé en jalons.
7. L'approbation du plan est `false` par défaut et reste optionnelle.

**Acceptation**
- Sur `examples/spec-example.md`, le plan contient au moins 2 steps, chacun
  avec `INTERFACES`, `TESTS`, `PITFALLS` et `DONE_WHEN` non vides.
- Une SPEC de 30 étapes produit des `MILESTONES` plus un lot de 12 steps au
  maximum.

### C7 — Audit high-tier actif : revue, refacto et correction en une passe

**Problème** : trois rôles se chevauchent :
- le reviser Claude, borné au scope ;
- le reviewer chat en lecture seule ;
- le cycle `REVIEW_IMPLEMENTATION` avec son `review_repair_planner`.

Aucun n'a le droit de refactorer. Chacun a son protocole, ses budgets et ses
états d'attente.

**Solution** : une seule phase `AUDIT`, exécutée par un **agent de code
high-tier** (`audit_profile`, par exemple `claude-opus-5-medium` ou
`codex-sol-medium`) :
1. **Entrées** :
   - la SPEC et le jalon courant ;
   - le plan du lot, avec les normalisations (C3) et les steps
     `MARK_FAILED` ;
   - le diff complet du lot ;
   - les `out_of_scope_paths` ;
   - le résultat du gate comparé à la baseline ;
   - les avertissements.
2. **Droits** : écriture sur tout le dépôt, sauf la hard-deny list. Il peut
   lancer les checks.
3. **Mandat**, dans l'ordre :
   - compléter ce qui manque pour le jalon ;
   - corriger les défauts ;
   - rendre le gate vert ;
   - supprimer le code mort et la rétrocompatibilité introduits ou rendus
     inutiles ;
   - simplifier les implementations.

   Un **budget de refacto** borne la dernière partie : ne pas toucher plus de
   `audit.max_refactor_paths` (15 par défaut) hors du diff du lot.
4. **Sortie** `META AUDIT v1` :
   - `STATUS: DONE | NEEDS_WORK | SPEC_DECISION` ;
   - `FIXED` ;
   - `REFACTORED` ;
   - `REMAINING` (liste de tâches concrètes pour le planner) ;
   - `RISKS`.
5. **Après l'audit**, le harness lance le gate :
   - s'il est rouge, l'audit reçoit l'extrait et fait jusqu'à 2 passes de
     réparation, ce qui absorbe la check-repair finale ;
   - s'il reste rouge, les échecs passent dans `REMAINING` et c'est le
     planner qui tranche (C8).
6. Un sous-mode `review_only` peut rester disponible pour un reviewer chat en
   lecture seule, mais il n'est plus dans le chemin par défaut.

**Supprime** :
- `semantic_revision.py`, `review_correction.py`, `review_recovery.py`,
  `candidate_review.py` (fusionnés en `audit.py`) ;
- les prompts `reviewer.txt`, `reviser.txt`, `review_repair_planner_v2.txt`
  et `check_repair.txt` (remplacé par l'audit pour le gate final) ;
- les phases `SEMANTIC_REVISION`, `FINAL_REVIEW`, `REVIEW_IMPLEMENTATION`,
  `REVIEW_REPLAN` et `CHECK_REPAIR` ;
- `ReviewVerdict` et `ReviewRoute`.

**Acceptation** :
- Faux lot avec une fonction morte et un test manquant : l'audit produit
  `DONE`, un commit d'audit et un gate vert.
- Faux lot irréparable : l'audit produit `NEEDS_WORK` avec `REMAINING` non
  vide, et le run **continue** (C8).

### C8 — Boucle planner ↔ audit (itérations)

**Solution** : la boucle de haut niveau du coordinateur devient :

```
context → PLAN(jalon suivant) → [IMPLEMENT step_i → gate rapide]* → GATE
       → AUDIT (+ réparations) → COMMIT d'itération
       → PLANNER_CONTINUE(SPEC, milestones, audit report, gate) → COMPLETE | NEXT_PLAN
       → … → PUBLISH
```

1. `PLANNER_CONTINUE` est un prompt court. Il reçoit la SPEC, les jalons,
   l'état de chaque jalon, le rapport d'audit (`REMAINING`, `RISKS`), le
   diffstat cumulé et les avertissements. Il renvoie :
   - `COMPLETE` ;
   - `NEXT` + `META PLAN v2` (un lot qui finit le jalon courant ou attaque le
     suivant) ;
   - `SPEC_DECISION`, qui est le seul chemin vers `WAIT_HUMAN`.
2. Chaque itération se termine par un commit sur la branche du run.
   Publication optionnelle à chaque itération (`publish.each_iteration`).
3. Garde-fous anti-boucle, qui mènent tous à `PUBLISH` en mode `PARTIAL` avec
   le rapport, et jamais à `WAIT_HUMAN` :
   - `max_iterations` (8 par défaut) ;
   - un budget global (C10) ;
   - détection de stagnation : 2 itérations consécutives avec le même
     ensemble `REMAINING` et un diff d'itération vide.
4. Le mécanisme remplace `CHECK_REPLAN`, `REPLAN_CYCLE`,
   `IMPLEMENTATION_CORRECTION_CYCLE`, les `CycleKind` de correction et
   `max_correction_cycles`.

**Supprime** : `planning/check_replan.py`, `orchestration/check_replan_service.py`,
`plan_recovery.py`, `orchestration/cycle_loader.py`,
`prompts/check_replan_planner_v2.txt`, `prompts/planner_correction_v2.txt`
(remplacé par `planner_continue.txt`).

**Acceptation** : une SPEC à 2 jalons avec un faux agent donne 2 itérations,
puis `COMPLETE`, puis `PUBLISHED`. Une stagnation forcée donne
`PUBLISHED(PARTIAL)` avec un rapport listant `REMAINING`.

### C9 — Reprise simplifiée : git fait autorité

**Solution**
1. Le checkpoint contient `{iteration, phase, step_index, last_green_commit,
   plan_sha256}`, et rien d'autre.
2. À la reprise :
   - `git reset --hard last_green_commit` dans le worktree, puis
     `git clean` limité au worktree ;
   - relecture du plan de l'itération ;
   - réexécution du step ou de la phase en cours depuis le début.
   L'idempotence vient de la réexécution, pas d'une preuve.
3. Les vérifications conservées : la branche du run existe,
   `last_green_commit` en est un ancêtre, et le hash du plan correspond au
   fichier.

**Supprime** : `orchestration/resume_integrity.py` (823 lignes),
`checkpoint_identity.py`, `orchestration/durable_readers.py`,
`orchestration/gate_acceptance.py`, `attempt_transaction.py` (remplacé par un
reset git), les chaînes `accepted-chain.json`, `RESUME_INTEGRITY_FAILURE`
(13 levées) et `RESUME_REQUIRES_OPERATOR`. Cible : `resume.py` ≤ 200 lignes.

**Acceptation** : un kill `-9` à n'importe quelle phase, suivi d'un `resume`,
donne un run qui termine, sur un test paramétré par phase.

### C10 — Un seul modèle de budget

**Solution** : remplacer les 12 budgets actuels
(`max_transient_attempts`, `max_executor_fallbacks`,
`max_check_infra_retries`, `max_review_transport_retries`,
`max_workspace_setup_retries`, `max_contract_repair_output_corrections`,
`max_contract_repair_planner_restarts`, `max_check_repair_attempts`,
`max_correction_cycles`, `max_step_contract_repairs`,
`max_preapproval_corrections`…) par :

```toml
[budget]
step_attempts = 3            # low, low+feedback, escalade tier
audit_repairs = 2
max_iterations = 8
max_wall_clock_hours = 12
max_cost_usd = 0             # 0 = illimité ; sinon somme usage.json
```

L'anti-boucle garde l'empreinte `(code, tree, facts)` : une même empreinte
passe au barreau suivant.

**Supprime** : `RecoveryBudgets` au profit de `Budget`, et la plupart des
champs de `RunOptions`.

### C11 — Suppressions et objectifs de taille

| Cible | Aujourd'hui | Objectif |
| --- | --- | --- |
| `src/` total | 48 527 | ≤ 32 000 |
| `orchestration/` modules | 36 | ≤ 14 |
| `RunPhase` | 17 | ≤ 10 (`CONTEXT, PLANNER, WORKTREE_SETUP, IMPLEMENT_STEP, GATE, AUDIT, PLANNER_CONTINUE, PUBLISH` + approbation optionnelle) |
| `RunStatus` | 23 | ≤ 10, dérivés de phase + disposition |
| Codes de failure distincts | ≈ 150 | ≤ 40 |
| Budgets de config | 12 | 5 |

Autres suppressions : le contract-repair dédié (`planning/contract_repair.py`,
`orchestration/contract_repair.py`, `orchestration/contract_recovery.py`,
environ 1 900 lignes). Un `META CONTRACT MISMATCH` d'exécutant devient
`REPLAN_STEP`, c'est-à-dire un appel planner court avec le contrat et
l'objection, qui renvoie un step de remplacement.

### C12 — Observabilité : la vérification humaine se fait après coup

**Solution** : un `iteration-NN/report.md` par itération, plus un `report.md`
final. Contenu :
- steps et tentatives, escalades, normalisations ;
- chemins hors scope, checks `SKIPPED_INFRA` et `baseline_red` ;
- rapport d'audit, coût et temps.

Ajouter un indicateur **autonomy score** : la part des runs terminés sans
`WAIT_HUMAN` ni `HARD_STOP`. Le web UI affiche ces rapports.

### C13 — Chaos suite d'autonomie (critère d'acceptation global)

**Solution** : `tests/autonomy/` utilise de faux agents et un faux LLM
scriptés. Chaque scénario doit finir en `PUBLISHED` ou `PUBLISHED(PARTIAL)`,
jamais en `WAIT_HUMAN` ni `HARD_STOP`. Scénarios :
1. rafale de 503 pendant 90 s sur le planner ;
2. CREATE sur un fichier existant ;
3. un worker qui édite un fichier hors scope ;
4. un worker qui commite ;
5. un worker qui ne change rien ;
6. un worker qui renvoie un protocole invalide ;
7. un test rouge à la baseline ;
8. le gate rouge après les steps, réparé par l'audit ;
9. le gate rouge irréparable, qui donne `PARTIAL` ;
10. un check d'intégration sans Docker, qui donne `SKIPPED_INFRA` ;
11. un code d'erreur inconnu levé au milieu du step ;
12. un kill -9 puis une reprise ;
13. une SPEC à 2 jalons.

Plus deux scénarios `FATAL` qui doivent s'arrêter : un secret dans le diff, et
une écriture dans `.github/workflows`.

**Et** 3 runs réels sur AutoWork après les lots A et B (§3). Le résultat est
consigné dans `docs/autonomy-runs.md`.

---

## 3. Ordre d'exécution

### Lot A — gains d'autonomie immédiats, peu de risque
Ce lot aurait sauvé les 5 runs réels observés.

| # | Chantier | Exécutant |
| --- | --- | --- |
| A1 | C0 HEAD vert | low-tier |
| A2 | C13 squelette de la chaos suite (scénarios 1, 2, 3, 4, 7, 10, 11 en `xfail`) | high-tier |
| A3 | C4 transport résilient + `WAIT_EXTERNAL` du planner + `--auto-resume` | low-tier |
| A4 | C3 normalisation des contrats | low-tier |
| A5 | C1 inversion de la politique | high-tier (conception) puis low-tier (tables) |
| A6 | C2 scope souple + reset des commits parasites | low-tier |
| A7 | C5 baseline + `SKIPPED_INFRA` + prompt check-repair | low-tier |

**Sortie du lot A** : les scénarios A2 passent et 1 run réel AutoWork aboutit.

### Lot B — remettre le flux dans la forme voulue

| # | Chantier | Exécutant |
| --- | --- | --- |
| B1 | C6 prompt planner + parser (sections riches, `MILESTONES`) | high-tier |
| B2 | C7 phase `AUDIT` (nouveau module, prompt, protocole) | high-tier |
| B3 | C8 boucle `PLANNER_CONTINUE` | high-tier |
| B4 | C10 budget unique | low-tier |
| B5 | C12 rapports | low-tier |

**Sortie du lot B** : les 13 scénarios C13 passent et 3 runs réels aboutissent.

### Lot C — simplification (après B uniquement)
Le lot B rend obsolètes les mécanismes que le lot C supprime.

| # | Chantier | Exécutant |
| --- | --- | --- |
| C-1 | Supprimer les phases et modules remplacés par C7 et C8 | low-tier, par module, avec la chaos suite comme filet |
| C-2 | C9 reprise git-first | high-tier |
| C-3 | C11 objectifs de taille ; supprimer contract-repair | low-tier |

### Règle de pilotage (A8)
Chaque prompt du lot est suivi d'une revue high-tier du diff avant le prompt
suivant. Aucun prompt n'est accepté s'il ajoute du code net sans chantier qui
le justifie. Un prompt qui ajoute un nouveau code de failure, un nouvel état
d'attente ou un nouveau budget est refusé, sauf si le chantier le demande
explicitement.

---

## 4. Décisions à trancher avant le lot A

1. **Hard-deny list par défaut.** Faut-il y mettre `Makefile`,
   `pyproject.toml` et les lockfiles ? Proposition : non pour
   `pyproject`/`Makefile`, qui sont admis et signalés à l'audit ; oui pour
   `.github/workflows`, `.env*` et la config MetaHarness.
2. **`MARK_FAILED_CONTINUE` contre arrêt.** Un step qui échoue après
   escalade : faut-il continuer les steps suivants qui en dépendent ?
   Proposition : non. On saute les steps dépendants (`DEPENDS_ON`), on
   exécute les autres, puis l'audit tranche.
3. **Publication.** Faut-il publier `PARTIAL` sur la branche du run ou ne
   rien publier ? Proposition : publier la branche avec une PR en draft et le
   rapport.
4. **Reviewer chat (bridge).** Faut-il le garder comme `review_only`
   optionnel ou le supprimer ? Proposition : le supprimer, puisque l'audit
   est un agent de code.
5. **`web/` et `remote/` (5 200 lignes).** Proposition : hors périmètre de
   v4, avec seulement l'adaptation aux nouvelles phases et aux nouveaux
   statuts.
