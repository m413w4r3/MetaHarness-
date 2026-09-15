# MetaHarness

MetaHarness V0 transforme un SPEC humain en un run Git contrôlé : un planner
produit un plan texte d’implémentation, Codex l’exécute dans un worktree isolé,
les checks déterministes figent les preuves, puis un reviewer compare SPEC,
plan et diff avant l’unique commit autorisé. L’implémenteur reçoit le plan,
pas le SPEC original ; le reviewer reçoit les deux. Une approbation humaine
optionnelle peut être exigée après le planner et avant la création du worktree.

Avec `[planning] protocol = "v2"` et `[revision] enabled = true` (le cas de
`examples/autowork.toml`), le pipeline complet est :

```text
main A
  ↓
isolated run worktree  (harness/<plan>/<run-id>, jamais le checkout AutoWork/)
  ↓
planner STAGED         (execution_mode_policy = "require-staged")
  ↓
Luna steps
  ↓
Claude                 (pre-checks → revision → final checks)
  ↓
reviewer #1
  ↓
optional C02           (repair planner → Luna repair → Claude C02 → reviewer #2)
  ↓
PASS
  ↓
commit B (parent = A)  (arbre exact approuvé, dans le worktree isolé)
  ↓
CAS fast-forward local main A→B   (git update-ref refs/heads/main B A)
  ↓
push origin/main A→B              (git push --porcelain origin B:refs/heads/main)
```

- maximum automatic cycles = 2 (C01 initial, C02 repair ; jamais de C03) ;
- `[revision]` fournit les defaults et le fallback legacy ; chaque nouveau run
  capture ses choix effectifs dans `run_options.json`. `claude_revision_enabled`
  et `repair_cycles` sont indépendants ; `repair_cycles` vaut actuellement
  seulement `0` ou `1` ;
- pour AutoWork, la portée de réparation recommandée est
  `repair_scope_policy = "auto-bounded"` avec
  `repair_scope_max_added_paths = 4` ;
- les agents ne travaillent jamais sur `main` : Luna, Claude et la review
  n’écrivent que dans le worktree isolé ; le checkout utilisateur n’est jamais
  modifié (ni checkout, ni index, ni fichiers) ;
- `[publish] mode = "fast-forward-base"` (AutoWork) : après le PASS final et le
  commit exact, `main` local avance par compare-and-swap de A vers B, puis B est
  poussé sur `origin/main`. Si `main` ou `origin/main` (ref de suivi locale, sans
  fetch implicite) a bougé : `BASE_MOVED_SINCE_RUN`, sans merge, rebase ni
  force. `mode = "run-branch"` pousse seulement la branche de run ;
- aucun force, lease, tag, delete ou merge automatique.

### Reprise : failure != lost work

Chaque transition durable met à jour `resume_checkpoint.json`, qui décrit
toujours la prochaine opération non encore réussie. Les phases supportées sont
conceptuellement : context, planner, plan approval, workspace setup, worker
steps, checks, Claude, reviewer, repair planner/steps, commit et publish.
Un run `failed` dont l’échec est reprenable
(Claude, Codex avant mutation, transport reviewer, push) se reprend au même
`run_id`, sans rejouer planner, approbation, setup ni step déjà réussi :

```bash
metaharness resume --config examples/autowork.toml --run-id <RUN_ID>
```

ou via le bouton unique de la page du run (`REPRENDRE À PARTIR DE CLAUDE`,
`RETRY S02`, `RETRY REVIEWER #1`, `RETRY PUBLISH`…). Avant toute reprise,
MetaHarness revérifie l’approbation, l’identité du plan, le hash de
l’execution selection, le worktree, la branche, HEAD, l’arbre candidat exact et
le scope approuvé ; au moindre écart : `RESUME_INTEGRITY_FAILURE`, sans aucun
appel LLM. Les corruptions, violations d’identité et
`AGENT_CONTRACT_MISMATCH` restent volontairement non-resumables.

Sans `revision.enabled`, le chemin historique reste disponible : un planner,
les steps Codex, les checks, un reviewer et un commit. Les
secrets ne sont jamais mis dans la configuration persistée : `api_key_env`
contient seulement le nom d’une variable d’environnement. Les écritures d’état
passent par `RunStateStore` et sont atomiques.

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
step Luna, la révision Claude, les checks, la review et la consommation de
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

## Vérification locale

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Pour l’architecture, les artefacts et l’exploitation, voir
[docs/architecture.md](docs/architecture.md), [docs/artifacts.md](docs/artifacts.md)
et [docs/runbook.md](docs/runbook.md).
