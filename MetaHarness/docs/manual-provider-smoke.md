# Smoke-test manuel réel du planner

Ce test contacte réellement le bridge ChatGPT UI et WebAI-to-API Gemini. Il
vérifie uniquement la chaîne HTTP provider → requête planner META PLAN v2 de
production (catalogue fixe `smoke-implementer` / `smoke-reviewer`, check
`test`) → Markdown généré → `OpenAIChatTextClient` → `parse_task_plan_v2()` →
plan `READY` ou `BLOCKED`.

Il ne crée pas de worktree, ne lance pas Codex, n'exécute pas les checks, ne
contacte pas le reviewer et ne committe rien. Il n'est pas exécuté par la CI
normale. Les réponses sont mesurées sans format repair.

## GPT UI

1. Démarrer le bridge ChatGPT UI.
2. Ouvrir ChatGPT.
3. Sélectionner manuellement dans l'UI le modèle voulu. MetaHarness ne choisit
   pas ce modèle ; le champ wire utilise toujours le label neutre
   `chatgpt-web`.
4. Exporter l'URL du bridge :

```sh
export META_SMOKE_GPT_BASE_URL="http://127.0.0.1:<PORT>"
```

Si le bridge demande une authentification :

```sh
export META_SMOKE_GPT_API_KEY="..."
```

Lancer :

```sh
python scripts/manual/planner_real_smoke.py chatgpt --repeat 3
```

## Gemini

1. Démarrer WebAI-to-API.
2. Exporter l'URL et l'identifiant réel du modèle stateless :

```sh
export META_SMOKE_GEMINI_BASE_URL="http://127.0.0.1:<PORT>"
export META_SMOKE_GEMINI_MODEL="<REAL_STATELESS_MODEL_ID>"
```

L'endpoint utilisé est le canonique `/v1/stateless/chat/completions`. Si le
service demande une authentification :

```sh
export META_SMOKE_GEMINI_API_KEY="..."
```

Lancer :

```sh
python scripts/manual/planner_real_smoke.py gemini --repeat 3
```

## Preflight

Avant toute génération, vérifier la configuration locale :

```sh
python scripts/manual/planner_real_smoke.py preflight
```

Le preflight ne génère aucun plan. Il vérifie les variables des deux
providers, la syntaxe des URLs, la validité des clés (par nom de variable,
jamais par valeur) et la lisibilité des fixtures. Pour Gemini, il interroge
uniquement `GET /v1/stateless/models` et confirme que
`META_SMOKE_GEMINI_MODEL` figure dans le catalogue. Pour ChatGPT UI, aucun
probe réseau n'est fait et le modèle n'est jamais sélectionné : il reste
choisi manuellement dans l'UI. Code retour : `0` prêt, `1` probe Gemini en
échec ou modèle absent, `2` configuration invalide.

## Matrice

Une fois les deux services prêts et leurs variables exportées :

```sh
python scripts/manual/planner_real_smoke.py matrix --repeat 3
```

Les artefacts sont écrits sous `manual-results/planner-smoke-<timestamp>/`
(ou sous le répertoire donné par `--output`). Chaque tentative contient la
requête exacte, la réponse Markdown exacte lorsqu'elle a été reçue, le plan
normalisé en cas de succès, et le résultat de la tentative. Une erreur de
parsing est écrite dans `parse-error.txt`. Une erreur de transport écrit dans
`result.json` son type (`error_type`) et un message `error` borné à 1000
caractères (HTTP 401, connexion refusée, timeout...), dont les valeurs de clés
configurées, en-têtes `Authorization`/`Cookie` et jetons bearer sont
masqués ; aucune traceback n'est persistée.

Le code retour vaut `0` si toutes les réponses reçues sont parsées et qu'il
n'y a aucun échec de transport, `1` sinon, et `2` si la configuration ou les
variables d'environnement sont invalides.

Ce lot ne lance pas automatiquement le test réel : l'opérateur doit démarrer
les deux services, ouvrir la session ChatGPT et fournir les credentials avant
d'exécuter les commandes ci-dessus.
