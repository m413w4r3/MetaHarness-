# Bridge Lab

Petite UI locale pour tester les deux services du dossier `Bridges` sans exposer
la clé du ChatGPT Bridge au navigateur.

## Utilisation dans le repo

Bridge Lab fait partie du compose principal de `MetaHarness-/Bridges`. Ajoute à
`.env.models` si tu veux changer le port :

```env
BRIDGE_LAB_PORT=7070
# Budget du polling background ; vide = BRIDGE_TOTAL_TIMEOUT + 120 s.
LAB_POLL_TIMEOUT_SECONDS=
```

Puis, depuis `MetaHarness-/Bridges`, construis et démarre la stack :

```bash
make models-build
make models-up
```

UI :

```text
http://127.0.0.1:7070
```

## Ce que le lab teste

- santé/readiness des deux services ;
- `/v1/models` ;
- `/v1/bridge/capabilities` côté ChatGPT Bridge ;
- `/v1/auth/status` et `/v1/runtime/status` côté WebAI ;
- Chat Completions ;
- Responses côté ChatGPT Bridge ;
- `/v1/bridge/runs` côté ChatGPT Bridge ;
- Chat stateless côté WebAI ;
- réponse SSE `stream=true` (bufferisée par le lab pour diagnostic) ;
- extraction du texte assistant ;
- métadonnées provider et erreurs de transport/parsing ;
- polling GET des Responses background jusqu'à l'état terminal, sans jamais
  refaire le POST initial ; borné par `LAB_POLL_TIMEOUT_SECONDS` (défaut :
  `BRIDGE_TOTAL_TIMEOUT` + 120 s) et interruptible par « Stop polling ».

## Sécurité

- les URLs provider sont codées côté serveur ; pas de proxy URL arbitraire ;
- `BRIDGE_API_KEY` reste seulement dans le conteneur `bridge-lab` ;
- `bridge-lab` n'est attaché qu'au réseau dédié `metaharness-bridge-lab` : ce
  proxy authentifié n'est pas joignable depuis `metaharness-models` (AutoWork) ;
- l'UI est publiée sur `127.0.0.1` ;
- tailles de requête/réponse bornées.

## WebAI : point important

Le catalogue `/v1/models` est informatif : le lab conserve `gemini-3-flash` comme
valeur initiale et affiche les modèles retournés comme suggestions éditables.
La readiness BrowserEngine/Playwright est affichée séparément de l'authentification
Gemini WebAPI ; un `/ready` à 503 ne rend donc pas automatiquement le WebAPI indisponible.
