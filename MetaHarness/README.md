# MetaHarness

MetaHarness transforme un SPEC humain en un run Git contrôlé : un planner
produit un plan texte d’implémentation, un backend sélectionné l’exécute dans un worktree isolé,
les checks déterministes figent les preuves, puis un reviewer compare SPEC,
plan et diff pour le candidat final avant publication. L’implémenteur reçoit le plan,
pas le SPEC original ; le reviewer reçoit les deux. Une approbation humaine
optionnelle peut être exigée après le planner et avant la création du worktree.

Avec `[planning] protocol = "v2"`, le pipeline complet et backend-neutral est :

```text
BASE
  ↓
isolated run worktree  (harness/<plan>/<run-id>, jamais le checkout AutoWork/)
  ↓
PLAN SINGLE or STAGED  (execution_mode_policy = "auto" by default)
  ↓
implementation steps → accepted step commits
  ↓
deterministic checks
  ├ FAIL → recovery ladder (check-repair, replan, fallback) → gate rerun
  └ PASS → semantic revision → deterministic checks
  ↓
accepted candidate D
  ↓
push exact D on the configured repository run branch
  ↓
read the remote tip and require tip == D
  ↓
final reviewer on immutable D
  ↓
PASS → publish exact reviewed SHA
REVISE / IMPLEMENTATION → semantic correction
REVISE / REPLAN → review repair planner
REVISE / HUMAN → operator required

approved candidate (fast-forward-base)
  ↓
CAS fast-forward local main A→B   (git update-ref refs/heads/main B A)
  ↓
push origin/main A→B              (git push --porcelain origin B:refs/heads/main)
  ↓
delete remote run branch           (fast-forward-base only; after publication)
```

- `max_check_repair_attempts` et `max_correction_cycles` sont deux budgets
  indépendants et configurables ; le second borne tout cycle après `INITIAL`,
  qu'il vienne d'une review ou d'un gate déterministe ; le reviewer ne connaît
  pas ces budgets ;
- une réparation de check corrige uniquement un signal déterministe ; une
  révision sémantique compare le candidat à la SPEC ; le reviewer final ne
  corrige jamais directement ;
- `REVISE / IMPLEMENTATION` réutilise le plan approuvé et appelle le reviser,
  tandis que `REVISE / REPLAN` appelle un planner correctif puis un nouvel
  implementer ; `REVISE / HUMAN` arrête toute correction automatique ;
- `[revision]` fournit les valeurs par défaut ; chaque nouveau run capture ses
  choix effectifs dans `run_options.json` ;
- pour AutoWork, la portée de réparation recommandée est
  `repair_scope_policy = "auto-bounded"` avec
  `repair_scope_max_added_paths = 4` ;
- les agents ne travaillent jamais sur `main` : les rôles sélectionnés
  n’écrivent que dans le worktree isolé ; le checkout utilisateur n’est jamais
  modifié (ni checkout, ni index, ni fichiers) ;
- `[publish] mode = "fast-forward-base"` (AutoWork) : chaque candidat exact est
  commité puis poussé sur la branche de run avant sa review ; après le PASS
  final, `main` local avance par compare-and-swap de A vers B, puis B est poussé
  sur `origin/main`. Une fois cette publication réussie, la branche distante
  temporaire du run est supprimée idempotemment. Si `main` ou `origin/main`
  (ref de suivi locale, sans fetch implicite) a bougé :
  `BASE_MOVED_SINCE_RUN`, sans merge, rebase ni force.
  `mode = "run-branch"` pousse seulement la branche de run ;
- `trace/events.v1.jsonl` est la preuve chronologique d’observation : plan,
  steps, checks, réparations, révisions, push, review et publication, avec les
  SHA/tree et les métadonnées de session disponibles ; il ne remplace jamais
  `RunStateStore` comme autorité ;
- aucun force, lease, tag ou merge automatique ; un échec du cleanup après
  publication laisse le run `PUBLISHED` avec un warning diagnostiqué.
- tous les candidats reviewables sont poussés sur la branche distante du run
  avant le reviewer final, même quand `publish.enabled = false`. Ce staging
  push est distinct de la publication post-PASS contrôlée par `[publish]` ;
  en `mode = "run-branch"`, `publish.remote` doit être `repository.remote`.

### Reprise : failure != lost work

Chaque transition durable met à jour `resume_checkpoint.json`, qui décrit
toujours la prochaine opération non encore réussie. Les phases supportées sont
conceptuellement : context, planner, plan approval, workspace setup, worker
steps, deterministic gates, reviewer, correction planner/steps, commit et publish.
Un run `failed` dont l’échec est reprenable
(avant mutation, transport reviewer, push) se reprend au même
`run_id`, sans rejouer planner, approbation, setup ni step déjà réussi :

```bash
metaharness resume --config examples/autowork.toml --run-id <RUN_ID>
```

ou via le bouton unique de la page du run (`REPRENDRE LE RUN`,
`RETRY S02`, `RETRY REVIEWER #1`, `RETRY PUBLISH`…). Avant toute reprise,
MetaHarness revérifie l’approbation, l’identité du plan, le hash de
l’execution selection, le worktree, la branche, HEAD, l’arbre candidat exact et
le scope approuvé ; au moindre écart : `RESUME_INTEGRITY_FAILURE`, sans aucun
appel LLM. Les corruptions, violations d’identité et
`AGENT_CONTRACT_MISMATCH` restent volontairement non-resumables.

Sans `revision.enabled`, la correction sémantique est désactivée : le planner,
les étapes d’implémentation, les checks, le reviewer et le commit restent
disponibles. Les
secrets ne sont jamais mis dans la configuration persistée : `api_key_env`
contient seulement le nom d’une variable d’environnement. Les écritures d’état
passent par `RunStateStore` et sont atomiques.

Les checks v2 proviennent du catalogue trusted `[[check_catalog]]` de la
configuration. Le planner ne sélectionne que leurs IDs dans
`REQUIRED_CHECKS`; les argv restent exclusivement dans MetaHarness. Les IDs
`default_check_ids` sont toujours requis, et les preflights configurés sont
exécutés avant les workers coûteux. Voir [docs/pipeline-v2.md](docs/pipeline-v2.md)
pour les invariants de chaîne, d’intégrité et de benchmark.

## Installation (une seule fois)

```bash
cd Bridges
cp .env.models.example .env.models   # seulement si absent
# remplir BRIDGE_API_KEY une fois
make models-up
make models-status

cd ../MetaHarness
/usr/bin/python3.12 -m pip install -e .

CODEX_HOME="$HOME/.local/share/metaharness/codex" codex login
CLAUDE_CONFIG_DIR="$HOME/.local/share/metaharness/claude" claude
metaharness doctor --config examples/autowork.toml
```

## Usage normal

```bash
metaharness doctor --config examples/autowork.toml

metaharness web \
  --config examples/autowork.toml \
  --port 8765
```

Après ce setup initial : aucun `export`, aucune copie de config, aucun
`PYTHONPATH`. Les chemins relatifs de `examples/autowork.toml` sont résolus
depuis `examples/` ; le secret `BRIDGE_API_KEY` vient uniquement de
`Bridges/.env.models`.

`doctor` est le gate local avant l’UI : config, fichiers d’environnement,
secrets utilisables (jamais affichés), repo propre et base SHA, racines de
runs/worktrees, binaires Codex et Claude Code, `CODEX_HOME` et
`CLAUDE_CONFIG_DIR` gérés sans MCP, runtime Codex managed, probe sandbox
`codex sandbox -- /bin/true`, probe de compatibilité CLI/parser Codex,
runtime Claude managed en mode restricted, probe parser-only Claude,
authentification locale Codex et Claude, exécutables de setup et de checks, et
`GET /health` du bridge local (127.0.0.1/localhost uniquement). Il ne lance
aucun modèle. Doctor doit être exécuté après l’authentification des runtimes
gérés.

Ouvrir `http://127.0.0.1:8765/`, cliquer sur `NEW RUN`, saisir le SPEC puis
`CREATE RUN`. Le planner, l’approbation des contrats de step exacts, chaque
step, la révision sémantique, les checks, la review et la consommation de
tokens sont ensuite suivis depuis la page du run.

La CLI reste disponible pour l’automatisation :

```sh
metaharness run --config examples/autowork.toml --spec examples/spec-example.md
```

Avec `[approval] require_plan_approval = true`, le run reste en
`awaiting_plan_approval` et aucun worktree n’est créé. Après inspection des
artefacts du plan, utiliser `approve-plan` ou `reject-plan` :

```sh
python -m metaharness.cli approve-plan --run ../MetaHarness-runs/example-001
# ou
python -m metaharness.cli reject-plan --run ../MetaHarness-runs/example-001
```

Une approbation reprend le pipeline ; un rejet termine le run en
`plan_rejected`. Le plan n’est pas éditable et une seconde décision est
refusée.

Les chemins TOML relatifs sont résolus depuis le fichier de configuration.
Les valeurs `${VARIABLE}` sont développées depuis l’environnement. Une
variable absente provoque une erreur. Le contrat provider est un POST
OpenAI-compatible texte, documenté dans [docs/providers.md](docs/providers.md).

## Remote Android gateway

Le gateway distant expose les routes d’observation et de contrôle de
MetaHarness à un client Android ; le port `--metaharness-port` est celui du
`metaharness web` local :

```bash
metaharness remote-gateway \
  --port 8770 \
  --metaharness-port 8765 \
  --remote-token-file ~/.config/metaharness/remote.token \
  --control-token-file ~/.config/metaharness/control.token
```

Les deux tokens sont lus avant tout bind, doivent être différents et ne sont
jamais affichés : le token distant authentifie le client du tunnel, le token de
contrôle reste le seul credential accepté par le MetaHarness local. Les ports
vont de 1 à 65535 et le port du gateway doit différer du port MetaHarness.

Chaque requête, `GET` compris, exige `Authorization: Bearer <token distant>`.
Les cinq routes de mutation n’acceptent qu’un corps JSON
(`Content-Type: application/json`, un seul `Content-Length`, aucun
`Transfer-Encoding`) dont les champs sont connus d’avance ; toute autre forme
est refusée avant le moindre appel local.

Le gateway écoute obligatoirement en localhost (`127.0.0.1`) : il n’existe
aucune option `--host` et il ne doit jamais être exposé directement. L’accès
distant passe par un tunnel privé tel que Tailscale Serve, qui ne publie
l’URL que dans le tailnet :

```bash
tailscale serve --bg 8770
```

Ne jamais utiliser Tailscale Funnel, et ne jamais ouvrir le port 8770
directement sur Internet : le gateway n’est joignable que depuis la machine
locale ou via le tunnel privé.

## Vérification locale

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m metaharness.cli config-check --config examples/autowork.toml
```

`python scripts/test_parallel.py [-j N] [module ...]` exécute les mêmes tests,
un processus par module, pour réduire le temps réel ; la commande `unittest
discover` ci-dessus reste la référence.

Pour l’architecture, les artefacts et l’exploitation, voir
[docs/architecture.md](docs/architecture.md), [docs/artifacts.md](docs/artifacts.md)
et [docs/runbook.md](docs/runbook.md).
