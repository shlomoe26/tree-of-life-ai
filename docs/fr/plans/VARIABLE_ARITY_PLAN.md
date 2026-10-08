# Test de généralisation : arité VARIABLE (écrit avant expérience)

*Suite de SAA_PARSER_PLAN.md. Attaque la limite documentée dans PAPER.md §6 : « fenêtre K+2 fixe,
non testé avec arité variable ». Décision : arité variable d'abord, domaine différent seulement si
jugé nécessaire après coup.*

## 1. Le problème technique

Le SAA-Parser (Phase 11) scanne des fenêtres de taille **fixe** `K+2` (opérateur + K opérandes +
CLOSE). Généraliser à une arité qui varie par nœud casse le tour différentiable utilisé pour
reconstruire la séquence (le décalage `R' = a·R[:-(K+1)] + y·C + b·R[K+1:]` suppose un retrait de
longueur constante à chaque étape). Reconstruire ce tour pour un retrait de longueur *variable par
exemple du batch* est un problème d'alignement non trivial (chaque exemple choisirait une fenêtre de
longueur différente au même pas, produisant des séquences résiduelles de tailles différentes).

## 2. La solution retenue : arité canonique Kmax avec padding masqué

- **Générateur** : arité *vraiment* variable par nœud, tirée dans `[Kmin=2, Kmax=7]`.
- **Sérialisation** : chaque groupe est *complété* à exactement `Kmax` opérandes avec un token
  **PAD** dédié (nouveau token de vocabulaire), de sorte que chaque nœud interne occupe toujours
  `Kmax+2` tokens bruts. On retrouve ainsi une fenêtre de taille fixe — **c'est le padding standard
  du deep learning** (comme un batch de séquences de longueurs différentes), pas une triche : le
  modèle doit *apprendre à ignorer* les emplacements vides, il ne les voit jamais « démasqués ».
- **Composeur (SAA)** : `SetAggCell` reçoit un masque de validité des enfants (nouveau paramètre) et
  met les logits d'attention des enfants PAD à `-inf` avant le softmax — masquage **explicite**, pas
  un espoir que le réseau apprenne à ignorer un embedding.
- **Scanner de fenêtres / scorer** : inchangé dans sa forme (fenêtre `Kmax+2`), mais doit aussi
  ignorer les PAD internes à la fenêtre pour le calcul K-aire lui-même.

C'est un compromis assumé : on ne teste pas une arité *illimitée*, mais une arité *variable et
inconnue à l'avance, jusqu'à un plafond* — le régime le plus réaliste (un vrai AST a une arité
variable mais rarement non bornée).

## 3. Implémentation (fichiers)

- `listops_data.py` : `gen_tree_var(depth, Kmin, Kmax, ...)` — arité tirée par nœud.
- `listops_model.py` : `SetAggCell.forward` accepte `child_mask` optionnel (True=valide).
- `listops_saa_parser.py` : nouveau mode de sérialisation avec padding à Kmax ; `gold_group_positions`
  adapté pour reconnaître PAD comme « déjà résolu » (comme une valeur) ; passage du masque enfant au
  composeur partout où il est appelé.
- **Test unitaire AVANT tout run long** (leçon des essais précédents — un bug de padding a déjà été
  détecté ainsi deux fois) : vérifier que (a) une arité 2 et une arité 7 dans le même batch donnent
  des résultats identiques à un run seul-exemple ; (b) le masque PAD est bien appliqué (permuter les
  PAD ne change pas la sortie).

## 4. Protocole (identique à la Phase 11 pour comparabilité directe)

Variante dure (SM, MED), Kmin=2, Kmax=7, cap nœuds/profondeur identiques, budget ~200k (ajusté pour
le padding), 10k iters + cosine + 50k données, 3 seeds. Modes : `sup` (structure gold) et `curr`
(curriculum, hold 30% puis sevrage — la config gagnante de la Phase 11).

## 5. Grille de lecture (pré-enregistrée)

| observation | conclusion |
|---|---|
| sup ≥ ~0,85 | le composeur K-aire+masquage gère l'arité variable ; diagnostic transfère |
| sup chute nettement (< 0,7) | le masquage de padding introduit une difficulté propre, pas juste l'arité |
| curriculum ≈ ratio Phase 11 (~98% de son propre plafond sup) | **le curriculum généralise** — résultat robuste, pas un artefact de K=5 fixe |
| curriculum ≪ ratio Phase 11 | le curriculum est sensible à la régularité de l'arité — limite réelle à documenter |

## 6. Calibration honnête

Le padding à Kmax n'est pas une généralisation *complète* à arité non bornée. Si ce test réussit, la
conclusion honnête est « le mécanisme généralise à arité variable bornée, avec masquage explicite » —
pas « à arité arbitraire ». Ce sera précisé dans RESULTS.md et PAPER.md quel que soit le résultat.
