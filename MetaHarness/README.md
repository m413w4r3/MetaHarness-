# MetaHarness

MetaHarness V0 transforme un SPEC humain en un run Git contrôlé : un planner
produit un plan texte d’implémentation, Codex l’exécute dans un worktree isolé,
les checks déterministes figent les preuves, puis un reviewer compare SPEC,
plan et diff avant l’unique commit autorisé. L’implémenteur reçoit le plan,
pas le SPEC original ; le reviewer reçoit les deux.

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
