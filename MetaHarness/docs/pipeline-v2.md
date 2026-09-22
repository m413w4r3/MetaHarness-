# Pipeline v2 : invariants opérationnels

MetaHarness est l’autorité d’exécution. Un run v2 est une machine d’état
durable : chaque frontière importante est matérialisée par `RunStateStore` et
par des artefacts atomiques. Les checks, les arbres Git et les SHA persistés
sont des preuves mécaniques ; aucun texte produit par un agent ne peut les
remplacer.

## Déroulement canonique

```text
PLAN
  ↓
implementation step
  ↓
accepted step commit
  ↓
...
  ↓
deterministic checks
  │
  ├ FAIL
  │   ↓
  │ bounded direct check-repair loop
  │   ↓
  │ deterministic checks
  │
  └ PASS
      ↓
    semantic revision
      ↓
    deterministic checks
      ↓
    accepted candidate
      ↓
    push run branch
      ↓
    final reviewer
      │
      ├ PASS
      │   ↓
      │ publish exact reviewed SHA
      │
      ├ REVISE / IMPLEMENTATION
      │   ↓
      │ semantic correction
      │
      ├ REVISE / REPLAN
      │   ↓
      │ review repair planner
      │
      └ REVISE / HUMAN
          ↓
        operator
```

Un step accepté ajoute un commit à la chaîne linéaire. Un essai rouge conserve
son arbre, son diff et son résultat de checks pour l’audit, mais ne devient
jamais un commit accepté. Une correction mécanique est donc toujours bornée
par `max_check_repair_attempts`. Une correction déclenchée par la review est
bornée séparément par `max_review_repair_cycles`.

Les checks déterministes sont exécutés avant chaque frontière qui peut
accepter un arbre. Une mutation du HEAD, de l’index, de l’arbre candidat ou
d’un état protégé est une erreur d’intégrité : le run se ferme sans nouvel
appel LLM/AgentExecutor pour « réparer » cette erreur.

Le candidat est un objet immutable. Il est poussé sur la branche du run avant
la review finale ; le reviewer reçoit ce SHA et non une branche mobile. Après
`PASS`, la publication réutilise exactement le SHA approuvé. Elle ne recrée
pas un commit et ne publie jamais un autre arbre.

## Responsabilités

- **check repair** corrige un signal déterministe précis (`lint`, `test`,
  `typecheck`, etc.). Il ne replanifie pas le produit et ne reçoit pas la
  responsabilité sémantique de la SPEC.
- **semantic revision** compare le candidat et les preuves à la SPEC. Elle
  peut corriger le comportement autorisé, puis doit repasser par les checks.
- **final reviewer** décide `PASS` ou une route corrective à partir d’un
  candidat immutable déjà vérifié et poussé. Il ne connaît pas le budget de
  correction ; le budget appartient au harness.

Les routes sont exclusives : `REVISE / IMPLEMENTATION` revient à la
correction sémantique ; `REVISE / REPLAN` passe par le planner de réparation de
review ; `REVISE / HUMAN` rend durable l’état `operator/human required` et
interdit toute correction ou publication automatique supplémentaire.

## Arbres différés et intégrité

`verification_status = deferred` autorise un commit intermédiaire seulement
si le contrat contient une raison, une commande/verification explicite et les
étapes futures dont dépend la vérification. Tant qu’une dépendance différée
n’est pas résolue par un step accepté, le candidat final est interdit.

Les échecs d’intégrité incluent notamment : HEAD inattendu, check qui modifie
le state protégé, secret détecté dans un blob staged et tree candidat différent
de la preuve. Ces cas sont fail-closed ; un planner ou un agent de correction
n’est jamais appelé pour les contourner.

## Profils et neutralité backend

Un profil sépare les axes suivants :

| Champ | Signification |
| --- | --- |
| `driver` / harness | exécuteur ou intégration locale qui lance le rôle |
| `provider` | fournisseur du modèle ou du service |
| `model` | identifiant du modèle choisi |
| `effort` | niveau d’effort demandé |
| `role` | responsabilité métier : planner, implementer, repair, reviser, reviewer |

Exemple conceptuel :

```text
driver=codex
provider=openai
model=gpt-5.6-luna
effort=high
role=implementer
```

`Luna` n’est qu’une valeur de profil. Aucun invariant ne suppose que le
semantic reviser est Claude, que le check repair est Codex ou que
l’implementer est Luna. Les tests v2 utilisent des fakes backend-neutral et
comparent le résultat métier, pas le nom du runtime.

## Compatibilité et observation

```text
MetaHarness
    = source of authority

META TRACE v1
    = stable observation contract

Nimbalyst
    = external cockpit / observer
```

Un artefact historique sans `pipeline_version` est interprété comme v1 et
reste dans la machine d’état legacy. Un nouveau run est v2 ; une reprise ne
peut jamais franchir cette frontière. META TRACE v1 observe les transitions,
les arbres, les SHA et les métriques disponibles sans devenir une seconde
autorité. Nimbalyst peut consommer cette trace comme cockpit externe, mais ne
peut pas autoriser un commit ou une publication.

## Matrice de benchmark

La matrice A/B doit conserver la SPEC, le plan, les checks, le budget, le
repository de départ et les fakes/outils constants. Seuls le profil ou le
backend comparé changent :

| Comparaison | Axe |
| --- | --- |
| Codex + Luna / Codex + DeepSeek | implémentation dans le même harness |
| Claude Code + DeepSeek / DeepSeek Harness + DeepSeek | harness/driver |
| Opus semantic reviser / DeepSeek semantic reviser | qualité de révision sémantique |

Mesures minimales :

- success/failure, nombre de repair attempts et review cycles ;
- wall time, prompt bytes, input/output tokens ;
- cache tokens et reasoning tokens lorsqu’ils sont disponibles ;
- tool count lorsqu’il est disponible ;
- nombre final de chemins modifiés et taille finale du diff.

Une métrique indisponible reste absente ou `null`, elle n’est pas inventée.
Les erreurs de configuration et les secrets ne font jamais partie des
résultats de benchmark.

## Validation locale

Depuis la racine du dépôt :

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m metaharness.cli config-check --config examples/autowork.toml
```

`config-check` est le contrôle de configuration fourni par la CLI. Les tests
utilisent `unittest` et des fakes locaux ; `pytest` n’est pas une dépendance
implicite.
