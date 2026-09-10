# MetaHarness model stack

Ce compose racine démarre `chatgpt-bridge` et `WebAI-to-API` sur le réseau Docker
partagé `metaharness-models`. Aucun gateway ou proxy intermédiaire n’est ajouté.

## Préparer les sous-modules

```bash
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

Ne placez pas de cookies Gemini dans les fichiers de configuration racine.

## Démarrer la stack

```bash
cp .env.models.example .env.models
```

Renseignez `BRIDGE_API_KEY` et `BRIDGE_WS_TOKEN` dans `.env.models`, puis :

```bash
make models-config
make models-up
make models-status
```

`models-down` arrête les services sans supprimer les volumes.

## Adresses

Depuis l’hôte :

```text
Bridge API: http://127.0.0.1:8001
WebAI API:  http://127.0.0.1:6969
```

Depuis un conteneur raccordé à `metaharness-models` :

```text
Bridge: http://chatgpt-bridge:8001/v1
WebAI:  http://web_ai:6969/v1
```

Les publications hôte restent limitées à loopback par défaut. Le volume
SQLite du Bridge est explicitement réutilisé sous le nom
`chatgpt-bridge_bridge_data`, et le runtime WebAI est conservé dans le chemin
défini par `WEBAI_RUNTIME_DIR`.

## Raccorder AutoWork

Après le démarrage de cette stack, configurer AutoWork avec :

```env
OPENAI_BRIDGE_BASE_URL=http://chatgpt-bridge:8001/v1
WEBAI_BASE_URL=http://web_ai:6969/v1
```

Puis, depuis AutoWork, utiliser son override réseau :

```bash
docker compose \
  -f compose.yaml \
  -f compose.models.yaml \
  up -d
```

Le réseau externe `metaharness-models` doit être déclaré par cet override ; la
stack racine en est propriétaire.
