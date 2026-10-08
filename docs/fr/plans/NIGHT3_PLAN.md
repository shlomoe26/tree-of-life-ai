# Nuit #3 — programme 7h+ carte blanche (écrit avant expérience)

Trois blocs enchaînés automatiquement (chaîne .bat avec attente sur checkpoint), GPU jamais au repos.

## Bloc 1 (en cours) — Arité variable

Voir `VARIABLE_ARITY_PLAN.md` (grille pré-enregistrée là-bas). sup+curr × 3 seeds, Kmin=2 Kmax=7.

## Bloc 2 — Attaque du problème ouvert : la chute OOD en profondeur (`--no_pos`)

**Hypothèse mécaniste (pré-enregistrée).** La Phase 11 laisse un problème ouvert : même sup/curriculum
perdent ~50% d'accuracy à profondeur 9. Suspect désigné : les **embeddings positionnels absolus
appris** — les séquences OOD sont plus *longues* (jusqu'à 233 tokens vs ~160 en train), donc les
positions hautes sont sous-entraînées et injectent du bruit. Or le SAA-Parser est **local** (scan de
fenêtres) : il n'a peut-être aucun besoin de positions globales.

**Test.** `listops_saa_parser.py --no_pos` (fixed-arity, variante dure SM+MED, config identique
Phase 11), sup+curr × 3 seeds. Comparaison directe aux chiffres Phase 11 (avec positions) :
sup 0,925/0,435 ood9 ; curr 0,909/0,459 ood9.

**Grille de lecture :**
| observation | conclusion |
|---|---|
| in ≈ pareil, ood9 ↑ nettement | **hypothèse validée** — les positions absolues étaient le poison OOD ; gain mécanistique sur le problème ouvert |
| in chute nettement | les positions étaient nécessaires au scoring (info d'ordre) — négatif propre |
| in ≈ pareil, ood9 ≈ pareil | les positions sont neutres ; la cause OOD est ailleurs (à documenter) |

## Bloc 3 — Domaine différent : opérateurs MODE et RNG

**Pourquoi ces opérateurs et pas la logique booléenne.** AND/OR/XOR sur {0,1} ne sont que des cas
particuliers de MIN/MAX/SM — un « nouveau domaine » factice. MODE (valeur la plus fréquente, égalité
→ plus petite) et RNG (étendue max−min) sont des **calculs d'une autre nature** : comptage de
fréquences et combinaison de deux statistiques d'ordre. Ni l'un ni l'autre ne se réduit aux 4 ops
d'origine. Chance = 0,1 (10 classes), comparable aux runs précédents.

**Test.** `listops_saa_parser.py --ops MODE,RNG` (fixed-arity), sup+curr × 3 seeds, config standard.

**Grille de lecture :**
| observation | conclusion |
|---|---|
| sup ≥ ~0,85 et curr ≈ ~95%+ de sup | le mécanisme (fenêtre K-aire + SAA + curriculum) **généralise à des sémantiques nouvelles** |
| sup haut mais curr décroche | le curriculum est sensible à la sémantique — limite réelle |
| sup bas (< 0,7) | MODE/RNG dépassent la capacité du composeur — limite du composeur, pas du parseur |

## Note de comparabilité

L'ajout de MODE/RNG au vocabulaire (NUM_OPS 4→6) décale les ids CLOSE/PAD et ajoute ~2 lignes
d'embedding (~128 params) à TOUS les entraînements futurs, y compris le bloc 2 (191 690 vs 191 562
params en Phase 11). Écart de budget négligeable (+0,07%), noté ici pour transparence.

## Dépouillement

Au fil de l'eau à chaque réveil horaire ; à la fin : RESULTS.md (Phases 12-14), PAPER.md
(section généralisation + mise à jour des Limites), graphe(s), mémoire, commits, rapport final —
avec jugement explicite : la généralisation est-elle suffisamment démontrée, ou reste-t-il un test
nécessaire ?
