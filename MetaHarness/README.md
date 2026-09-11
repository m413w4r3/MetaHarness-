# MetaHarness

MetaHarness V0 transforme un SPEC humain en un run Git contrôlé : un planner
produit un plan texte d’implémentation, Codex l’exécute dans un worktree isolé,
les checks déterministes figent les preuves, puis un reviewer compare SPEC,
plan et diff avant l’unique commit autorisé. L’implémenteur reçoit le plan,
pas le SPEC original ; le reviewer reçoit les deux. Une approbation humaine
optionnelle peut être exigée après le planner et avant la création du worktree.

V0 reste volontairement simple : un seul task, un seul agent, un seul planner,
un seul reviewer, sans multi-agent, repair loop ni comportement mock. Les
secrets ne sont jamais mis dans la configuration persistée : `api_key_env`
contient seulement le nom d’une variable d’environnement. Les écritures d’état
passent par `RunStateStore` et sont atomiques.

## Démarrage rapide

```sh
cp examples/autowork.toml autowork.local.toml
python -m metaharness.cli config-check --config autowork.local.toml
python -m metaharness.cli run \
  --config autowork.local.toml \
  --spec examples/spec-example.md \
  --run-id example-001
```

Pour le flux normal depuis l’UI locale, démarrer le serveur puis ouvrir
`http://127.0.0.1:8765/` :

```sh
metaharness web --config autowork.local.toml
```

Cliquer sur `NEW RUN`, saisir le SPEC puis `CREATE RUN`. Le planner, la
demande d’approbation éventuelle, Codex, les checks et la review sont ensuite
suivis depuis la page du run.

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

Le profil exemple est copié à la racine car les chemins TOML relatifs sont
résolus depuis le fichier de configuration. Ne versionnez pas la copie locale.
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
