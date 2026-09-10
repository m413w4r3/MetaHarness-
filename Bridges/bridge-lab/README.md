# Bridge Lab

Petite UI locale pour tester les deux services du dossier `Bridges` sans exposer
la clé du ChatGPT Bridge au navigateur.

## Installation dans le repo

Depuis `MetaHarness-/Bridges` :

```bash
cp -R /chemin/vers/bridge-lab ./bridge-lab
cp /chemin/vers/compose.bridge-lab.yaml ./compose.bridge-lab.yaml
```

Ajoute à `.env.models` si tu veux changer le port :

```env
BRIDGE_LAB_PORT=7070
```

Puis démarre la stack existante + le lab :

```bash
docker compose \
  --env-file .env.models \
  -f compose.models.yaml \
  -f compose.bridge-lab.yaml \
  up -d --build
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
- parsing JSON strict de la réponse entière ;
- détection des code fences ;
- validation JSON Schema 2020-12.

Le test "JSON strict" n'enlève volontairement pas les ``` ni le texte parasite.
Il permet donc de voir si une sortie passerait le comportement actuel d'AutoWork,
qui attend un document JSON complet avant validation Pydantic.

## Sécurité

- les URLs provider sont codées côté serveur ; pas de proxy URL arbitraire ;
- `BRIDGE_API_KEY` reste seulement dans le conteneur `bridge-lab` ;
- l'UI est publiée sur `127.0.0.1` ;
- tailles de requête/réponse bornées.

## WebAI : point important

La version actuellement épinglée de WebAI-to-API rejette `response_format` pour
Gemini. Les presets JSON du lab reposent donc uniquement sur l'instruction dans
le prompt et valident ensuite le résultat localement. Cela permet de mesurer
directement si Gemini renvoie un JSON strict utilisable par AutoWork.
