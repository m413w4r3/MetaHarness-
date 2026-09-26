# Runs réels AutoWork (Lot A)

Un run est consigné ici uniquement lorsqu'il a réellement tourné contre des
providers externes. Un `NOT_RUN` n'est jamais un PASS.

## Modèle d'entrée

```text
date:
HEAD:
spec/run target:
result:
final status:
WAIT_HUMAN count:
HARD_STOP count:
retries/escalations:
gate baseline warnings:
failure reason if any:
```

## 2026-09-26 — baseline gates (prompt A7)

```text
date: 2026-09-26
HEAD: 55cbf1b (worktree non commité)
spec/run target: Lot A / scénarios 01, 02, 03, 04, 07, 10, 11
result: NOT_RUN
final status: NOT_RUN
WAIT_HUMAN count: NOT_RUN
HARD_STOP count: NOT_RUN
retries/escalations: NOT_RUN
gate baseline warnings: NOT_RUN
failure reason: aucun provider/credential réel disponible dans
  l'environnement d'exécution (pas d'accès réseau sortant vers les bridges
  provider, pas de clé exportée) ; aucun fake n'a été substitué au run réel.
```

Les scénarios automatisés du Lot A (01, 02, 03, 04, 07, 10, 11) restent
couverts par `tests/autonomy/` et ne dépendent pas de ce run manuel.
