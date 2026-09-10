# MetaHarness

MetaHarness V0 fournit les fondations strictes de l'orchestration future :
configuration TOML validée, modèles immuables et état de run JSON écrit
atomiquement.

## Vérifier une configuration

```sh
python -m metaharness.cli config-check --config config.toml
```

Les valeurs `${VARIABLE}` sont développées depuis l'environnement. Une
variable absente provoque une erreur. Les clés API ne sont jamais affichées ni
écrites dans l'état d'un run : `api_key_env` désigne uniquement le nom de la
variable d'environnement.

## Tests

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```
