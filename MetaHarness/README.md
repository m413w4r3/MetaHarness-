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
`CLAUDE_CONFIG_DIR` gérés sans MCP, probe `codex sandbox -- /bin/true`,
capacités `claude --help`, authentification locale Codex et Claude,
exécutables de setup et de checks, et `GET /health` du bridge local
(127.0.0.1/localhost uniquement). Il ne lance aucun modèle. Doctor doit être
exécuté après l’authentification des runtimes gérés.

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
