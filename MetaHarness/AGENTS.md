# MetaHarness — instructions de dépôt

- Python >= 3.12 et bibliothèque standard autant que possible.
- Aucun appel réseau dans le bootstrap V0.
- Ne jamais importer `references/agent_runner.py` dans le code de production.
- Les écritures d'état passent par `RunStateStore` et sont atomiques.
- Ne jamais stocker ou afficher de secret ; `api_key_env` contient un nom de
  variable d'environnement, jamais sa valeur.
- N'appel pas les test adversariaux (trop lent)
