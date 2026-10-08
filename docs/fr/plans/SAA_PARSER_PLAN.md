# SAA-Parser — plan de conception (écrit AVANT toute expérience)

*Le swing final du projet : la synthèse des deux frontières, désignée par les données.*

## 1. D'où vient cette architecture (le raisonnement, pas l'élégance)

Deux faits établis par nos expériences :

1. **Frontière B (réussie)** : l'agrégation **K-aire apprise** (SAA, attention sur ensemble) calcule
   nos opérateurs à 0,988 sur arbre gold — quand toute cellule binaire plafonne à ~0,45.
2. **Frontière A (diagnostiquée)** : l'induction de structure échoue **non pas** parce que la structure
   est inapprenable (supervisée, le parsing libre rejoint son plafond : 0,411 ≈ 0,454), mais parce que
   **toute composition binaire bute sur le même mur ~0,42-0,46** — MED n'est pas décomposable par
   paires, le fold binaire devrait porter le multiset entier.

**La synthèse désignée** : un parseur à structure latente qui fusionne des **groupes K-aires entiers**
(un opérateur + son span d'opérandes) en un coup, avec la **SAA comme composeur**. Personne (à notre
connaissance, dans notre périmètre) ne fait du parsing latent K-aire à agrégation apprise — le SOTA
(BT-GRC, beam search) est binaire.

## 2. L'hypothèse centrale (« œuf et poule »), pré-enregistrée

L'induction de structure non supervisée échouait peut-être *parce que* le composeur binaire ne savait
rien calculer (plafond 0,45) : bonne ou mauvaise fusion, la loss ne bougeait presque pas → **aucun
signal** pour apprendre la structure. Avec un composeur K-aire qui *sait* calculer (0,99), une bonne
fusion paie immédiatement → le signal de structure devient informatif → **les deux moitiés
s'amorcent mutuellement**.

## 3. L'architecture

Séquence plate de tokens (notation préfixe : `op c1 … cK ]`). Boucle de réduction :

1. **Scorer de span** : pour chaque position i, un score s_i = « la fenêtre [i .. i+K+1] est-elle un
   groupe réductible (op, K opérandes, CLOSE) dont les opérandes sont déjà des feuilles/valeurs ? ».
   (K est fixe = 5 dans notre générateur → fenêtre de taille fixe K+2 ; limitation annoncée, comme
   pour l'arité du générateur.)
2. **Sélection straight-through** (softmax ST comme pour le parseur binaire — le mécanisme de
   sélection n'était PAS le problème, on le garde).
3. **Composition K-aire** : le groupe sélectionné est réduit en UN nœud par la **SAA**
   (op → embedding opérateur, les K opérandes → children de la SetAggCell). Le token CLOSE est
   consommé. La séquence raccourcit de K+1.
4. Répéter jusqu'à 1 nœud → tête de classification.

Reconstruction différentiable de la séquence : même principe de masquage cumulatif que le parseur
binaire (généralisé au retrait d'un bloc de K+1 positions).

## 4. Protocole (identique aux runs précédents → comparaisons directes)

Variante dure (SM, MED), K=5, cap 120 nœuds, profondeurs train 1-4 / OOD 5-7-9, 50k exemples,
10k iters + cosine, batch 64, 3 seeds, data_seed 1234, budget ~200-250k params.

Deux étapes, dans l'ordre :
- **Étape 1 — validation du diagnostic (structure supervisée)** : teacher forcing des réductions
  gold. Si le plafond `in_tf` explose au-delà de 0,45 (vers ~0,9+), le diagnostic « la binarisation
  était le mur » est **prouvé causalement**. Si in_tf reste ~0,45, notre diagnostic est faux (et il
  faudra le dire).
- **Étape 2 — le vrai test (structure libre)** : parsing non supervisé. C'est le test de l'hypothèse
  œuf-et-poule.

## 5. Grille de lecture (pré-enregistrée, à ne pas déplacer)

| observation | conclusion |
|---|---|
| Étape 1 : in_tf ≥ ~0,9 | diagnostic binarisation **prouvé** ; continuer |
| Étape 1 : in_tf ~0,45 | diagnostic **réfuté** — le mur est ailleurs ; s'arrêter et le documenter |
| Étape 2 : libre ≥ ~0,85 (≈ in_tf) | **hypothèse œuf-poule validée — architecture innovante démontrée** : premier parseur latent K-aire à agrégation apprise battant toute composition binaire |
| Étape 2 : libre ≫ 0,47 mais < in_tf | structure partiellement apprise : gain réel, co-apprentissage incomplet |
| Étape 2 : libre ≈ 0,45-0,47 | œuf-poule réfutée : même avec un bon composeur, pas de signal — le co-apprentissage est le mur suivant |

Toutes les issues sont publiables ; les deux dernières ferment proprement.

## 6. Risques identifiés d'avance

- La fenêtre K+2 fixe est une simplification forte (assumée, comme l'arité du générateur).
- Le scorer doit apprendre « mes opérandes sont des feuilles » — s'il se trompe tôt, erreurs en cascade
  (même exposure bias que le binaire). L'étape 1 le neutralise (teacher forcing), l'étape 2 le mesure.
- Attention aux bugs de padding/batch (leçon de l'essai #3 : bug des racines courtes — vérifier avec
  un test unitaire AVANT tout run long).

## 7. Calibration honnête (répétée pour ne pas s'auto-hypnotiser)

Même en cas de succès complet : innovation **démontrée sur une tâche de niche**. Pour « compter »
au-delà, il faudrait transférer (autres tâches, arité variable, vraie ListOps). Mais « premier parseur
latent K-aire à agrégation apprise » + le diagnostic mécaniste qui l'a désigné serait un résultat
authentiquement nouveau — le premier du projet à mériter ce mot.
