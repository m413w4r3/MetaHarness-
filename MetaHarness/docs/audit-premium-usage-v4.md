# Usage des modèles premium — audit du run 20260928T135523Z-85b255f156

## Ce que les artefacts établissent

La limite Claude est celle d'une fenêtre de consommation de session. Elle ne
prouve ni cinq heures de travail continu sur cette tâche, ni que tout le quota
de la fenêtre a été consommé par ce seul run. La passe examinée dure environ
18 minutes. Aucun montant facturé n'est déduit des compteurs de tokens.

| Mesure de la passe d'audit interrompue | Valeur |
| --- | ---: |
| Prompt initial | 706 678 octets |
| Diff injecté, sérialisation JSON | 534 938 octets |
| Plan injecté, sérialisation JSON | 76 014 octets |
| Baseline puis verdicts de baseline répétés | 11 065 + 10 545 octets |
| Steps du jalon | 14 |
| Tours Claude | 259 |
| Tokens de sortie déclarés | 100 420 |
| Tokens lus depuis le cache déclarés | 84 956 817 |
| Appels d'outils | 69 Read, 73 Grep, 2 Glob, 114 Edit |
| Fichiers modifiés | 33 : 13 fichiers source, 20 fichiers de tests |
| Prompts des workers | 38 459–42 597 octets ; médiane 41 145,5 |

Ces valeurs proviennent du prompt, du flux Claude et des artefacts sauvegardés
dans `manual-recovery/20260928T220904Z-audit-fallback/`. Le plan JSON injecté
contenait également le texte brut du plan, en plus de ses champs structurés.
Le cache ne rend pas ces tokens gratuits ; sa tarification et son poids dans
le quota ne sont pas estimés ici.

Les lectures de fichiers n'étaient pas principalement des lectures intégrales :
67 des 69 Read avaient une limite, et aucune requête Read identique n'a été
répétée. Le problème confirmé est surtout l'injection systématique du gros
diff et des contrats, puis la multiplication des tours de réparation.

## Pourquoi l'audit a pris en charge autant de travail

Les 14 steps ont été acceptés, mais les checks avant commit étaient seulement
`lint` et `typecheck`. « COMPLETED » n'établissait donc pas que les tests
comportementaux ou d'intégration étaient verts. Le gate final rapporte :

- un échec de collecte des tests unitaires : `test_production_reuse.py`
  importe encore `_synthesis_input_hash`, supprimé dans le diff de S13 ;
- 35 échecs d'intégration, avec 167 tests d'intégration verts ;
- des corrections relevées dans le flux d'audit : ligne de read model sans
  `publication_language`, double de ModelGateway non adapté à la nouvelle
  requête de draft, fixtures de contenu et contrats de réparation obsolètes.

Il existe donc des omissions concrètes dans le travail livré. Cela ne justifie
pas de qualifier tous les workers de mauvais : 13 steps ont été acceptés au
premier essai et S03 au troisième, sur un contrôle encore insuffisant pour
détecter ces omissions. Le plan de S13 n'incluait pas tous les importers de la
fonction supprimée ; lint/typecheck n'ont pas compensé ce trou.

Le jalon M01 cumulait domaine, stockage, grounding, draft, réutilisation,
révision, workflow, invalidation, compatibilité Assembly, suppression de l'état
de conversation et preuve d'intégration. L'intitulé commun « backend Synthesis »
masquait plusieurs responsabilités indépendamment vérifiables. Le prompt
d'audit demandait en outre de compléter les comportements manquants de toute
la SPEC sans rappeler explicitement la frontière avec M02.

## Corrections appliquées

Le prompt partagé par Claude et Sol conserve la SPEC complète, le seul jalon
courant, ses attendus, contraintes, risques, tâches restantes et verdicts du
gate. Il ne colle plus le diff, les contrats de tous les steps, leurs réponses
finales ni la baseline entière. Les sources restent dans le worktree.

Le diff complet est disponible en `diff.patch`, avec des références aux contrats
et aux logs. Claude, dont ce runtime interdit le shell, reçoit un accès de
lecture à ces artefacts : `--add-dir` avec permissions de lecture et interdiction
explicite d'Edit/Write sur le répertoire du run. Les commandes et outils de
réseau restent restreints. Sol peut aussi consulter les artefacts ou le diff
Git. Le diff du fallback est actualisé avec les modifications de son prédécesseur
et expurgé avant persistance.

Chaque jalon utilise l'arbre précédant son premier step, afin de ne pas
réauditer les diffs des jalons déjà terminés. Les logs persistés sont maintenant
résolus correctement après reprise (`stdout_log_path` / `stderr_log_path`),
en complément des champs des résultats vivants.

`prompt_budget.audit_max_bytes` vaut 64 000 octets. Les extraits de logs sont
secondaires et peuvent être raccourcis ; la SPEC, les limites du jalon et les
verdicts ne sont jamais tronqués. Une autorité plus grande que le budget
produit un dépassement explicite dans les diagnostics. Claude et chaque
fallback ont leurs propres diagnostics de taille.

Une reconstitution complète de l'entrée historique donne **environ 60 Ko**, soit
**−91,5 %** par rapport aux 706 678 octets initiaux, sans troncature d'autorité.
Ce résultat mesure l'entrée ; il ne constitue pas une mesure de facture ou
une garantie de baisse proportionnelle du coût total d'une future session.

Les prompts planner et continuation exigent des jalons plus étroits, des
migrations atomiques avec tous leurs callers et fixtures, et des vérifications
ciblées exécutables. L'exemple limite les futurs plans à six steps et ajoute
`test-collection` au gate avant commit : collecte pytest, sans exécuter les tests
ni leurs fixtures PostgreSQL. La régression d'import est ainsi rendue au worker
avant le premium. Cela ne remplace pas les tests comportementaux ou d'intégration.

L'auditeur mène une passe de réparation centrée sur son jalon et renvoie les
travaux plus larges dans `NEEDS_WORK/REMAINING` pour le planner et les workers,
au lieu d'implémenter les jalons suivants. Le gate final reste obligatoire.
Les options et checks des runs déjà approuvés restent liés à leurs snapshots ;
le nouveau découpage et le nouveau catalogue concernent les futurs runs.

## Vérification et état du run

Les tests couvrent le budget et la conservation de l'autorité, les logs après
reprise, la collecte d'un import supprimé corrigée par le worker avant l'audit,
la frontière entre deux jalons, le diff actualisé du fallback et ses diagnostics,
ainsi que la transmission des permissions de lecture à Claude. Le parseur du
Claude installé accepte les arguments de lecture des artefacts avec `--help`.
La suite ciblée passe : **129 tests**. Ruff passe sur les deux nouveaux modules
Python ; la comparaison avec HEAD ne montre aucun nouveau diagnostic dans les
fichiers existants modifiés. `git diff --check` passe également.
Aucun appel de modèle ni run réel n'a été relancé pour cette vérification.

Le run réel est désormais `failed/RESUME_INTEGRITY_FAILURE` : HEAD pointe sur
`538a119` (« claude audit »), que la reprise ne reconnaît pas comme commit
de cette autorité. Ce commit n'a été ni réécrit ni adopté pendant cette tâche.
