# MetaHarness model stack

Le compose commun démarre `chatgpt-bridge`, `WebAI-to-API` et `bridge-lab` sur
le réseau Docker partagé `metaharness-models`. Les trois services peuvent donc
se joindre par leurs noms `chatgpt-bridge`, `web_ai` et `bridge-lab`. Aucun
gateway ou proxy intermédiaire n’est ajouté.

## Préparer depuis la racine MetaHarness-

Depuis la racine du dépôt :

```bash
cd Bridges
git submodule update --init --recursive
cd WebAI-to-API
python scripts/bootstrap.py
cd ..
```

Le bootstrap crée les fichiers locaux `.env`, `config.conf` et `runtime/` de
WebAI sans les versionner. En cas de préparation manuelle :

```bash
cd WebAI-to-API
cp .env.example .env
cp config.conf.example config.conf
mkdir -p runtime
cd ..
```

Ne versionnez jamais `config.conf`, `.env`, les cookies Google ou
`runtime/auth/*`. Ne placez pas de cookies Gemini dans les fichiers de
configuration racine.

## Cycle de vie

Toujours exécuter les commandes depuis `MetaHarness-/Bridges` :

```bash
cp .env.models.example .env.models
```

Renseignez `BRIDGE_API_KEY` et `BRIDGE_WS_TOKEN` dans `.env.models`, puis :

```bash
make models-volume
make models-build
make models-up
make models-status
```

Le volume SQLite historique est externalisé sous le nom
`chatgpt-bridge_bridge_data`. `models-up` et `models-rebuild` l’inspectent et
le créent seulement s’il n’existe pas ; cette préparation est idempotente.
`models-volume` peut donc aussi être exécutée seule pour préparer une
installation neuve.

- `models-build` construit explicitement les images et peut donc nécessiter
  l’accès aux registres (notamment l’image Playwright de WebAI).
- `models-up` utilise les images locales existantes, sans rebuild ni pull
  implicite.
- `models-rebuild` construit explicitement, puis démarre avec ces images.
- `models-down` arrête les services sans `-v` et conserve les volumes.
- `models-status` affiche `ps` ; `models-logs` suit les logs récents.

Ne lancez jamais `docker compose down -v` pour cette stack : le volume peut
contenir `bridge-runs.sqlite3`.

## Validation

La configuration effective peut être vérifiée ainsi :

```bash
docker compose --env-file .env.models -f compose.models.yaml config
```

Les probes minimales sont :

```bash
curl -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/health
curl -o /dev/null -w '%{http_code}\n' http://127.0.0.1:6969/health
curl http://127.0.0.1:6969/v1/auth/status
```

`/ready` de WebAI peut rester en 503 en mode Gemini WebAPI pur, même lorsque
Gemini est fonctionnel et authentifié. Ce probe ne doit donc pas être utilisé
seul pour valider l’authentification Gemini WebAPI.

## Adresses

Depuis l’hôte, les publications restent limitées à loopback :

```text
Bridge API: http://127.0.0.1:8001
WebAI API:  http://127.0.0.1:6969
Bridge Lab: http://127.0.0.1:7070
```

Depuis un conteneur raccordé à `metaharness-models` :

```text
Bridge: http://chatgpt-bridge:8001/v1
WebAI:  http://web_ai:6969/v1
Lab:    http://bridge-lab:7070
```

Les bind mounts conservent la configuration WebAI (`config.conf` en lecture
seule) et l’état d’authentification/runtime (`WEBAI_RUNTIME_DIR`).

## Raccorder AutoWork

Après le démarrage de cette stack, configurer AutoWork avec :

```env
OPENAI_BRIDGE_BASE_URL=http://chatgpt-bridge:8001/v1
WEBAI_BASE_URL=http://web_ai:6969/v1
```

Puis, depuis `MetaHarness-/AutoWork`, utiliser son override réseau :

```bash
docker compose \
  -f compose.yaml \
  -f compose.models.yaml \
  up -d
```

Le backend et le worker AutoWork concernés rejoignent le réseau externe
`metaharness-models`, comme les services de cette stack, et résolvent donc les
noms `chatgpt-bridge` et `web_ai`.
