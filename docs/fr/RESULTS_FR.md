# Expérience ListOps récursive — l'Arbre nourri de *vraie* structure

## La question

Sur du texte, l'archi Etz HaChaim perdait contre un Transformer **et** ajouter des mondes
ne changeait rien (E0 ≈ E4). Diagnostic : le code recopiait un seul vecteur sur les 10
Sefirot (`x_nodes = x_flat.repeat(1,10,1)`) — la structure n'entrait jamais par la donnée.
C'était un MLP déguisé.

**Le renversement testé ici :** l'Etz n'est plus le réseau qui lit toute l'expression, c'est
la **cellule appliquée à chaque nœud** d'un arbre de syntaxe ListOps. L'opérateur entre dans
S0, chaque opérande dans S1..S5, le reste = registres brouillon, lecture du résultat sur
Malchut (S9). Pour la première fois, de **vraies pièces distinctes** entrent dans les Sefirot.

## Protocole (garde-fous pré-enregistrés)

- **Tâche** : ListOps (MAX/MIN/MED/SM mod 10), arité fixe K=5, sortie = 1 chiffre (classif 10).
- **Cap 120 nœuds** : borne la mémoire ET isole la PROFONDEUR comme seule variable du test OOD.
- **Budget égal** : R0 vanille 1,127M ≈ Etz1 1,076M ≈ Etz4 1,111M.
- **Garde-fou 1 (calibration)** : R0 ne sature pas in-dist (0,745) → tâche bien réglée.
- **Garde-fou 2 (juge n°1)** : extrapolation profondeur (ood5/6/7 = plus profond que l'entraînement 1–4).
- **3 seeds**, mêmes données, 3000 itérations, AdamW lr 1e-3.

## Résultats (accuracy, moyenne ± écart-type sur 3 seeds)

| modèle | in (≤4) | ood5 | ood6 | ood7 |
|---|---|---|---|---|
| **R0** (vanille) | **0.745 ± 0.009** | 0.698 ± 0.014 | 0.711 ± 0.007 | 0.757 ± 0.007 |
| **Etz1** (1 monde) | 0.536 ± 0.001 | 0.504 ± 0.003 | 0.520 ± 0.004 | 0.523 ± 0.009 |
| **Etz4** (4 mondes + Tzimtzum) | 0.664 ± 0.008 | 0.638 ± 0.004 | 0.652 ± 0.011 | 0.674 ± 0.013 |

## Verdict (selon la grille fixée AVANT de voir les chiffres)

**Résultat nul *nuancé* — branche « Etz ≤ vanille ».**

1. **Etz4 ≫ Etz1 (+0.13, robuste).** La malédiction « E0 ≈ E4 » du texte est **brisée** : quand
   la structure entre vraiment, la profondeur hiérarchique + le Tzimtzum apportent quelque chose.
   → Les *principes* (hiérarchie, bottleneck) sont partiellement validés.

2. **Mais Etz4 < R0 partout (~−0.07).** L'architecture complète ne bat **pas** une cellule
   vanille de même budget. Elle « gagne contre elle-même affaiblie », pas dans l'absolu.
   → L'architecture ne **gagne pas son loyer**. La Voie B (vrais graphes) n'est pas justifiée par ce test.

3. **Le juge OOD ne discrimine pas.** R0 ne se dégrade pas en profondeur (in 0.745 → ood7 0.757) :
   la cellule plate **extrapole déjà bien**. Il n'y a aucun trou d'extrapolation que la hiérarchie
   pourrait combler. L'écart Etz4↔R0 est ~constant à toute profondeur.

## Conclusion honnête

On valide les **intuitions** que la Kabbale a inspirées (la hiérarchie aide sur des données
structurées), **pas** la cosmologie : 10 Sefirot, 22 chemins, 4 mondes restent non motivés par
la tâche, et la mise en œuvre kabbalistique perd contre un baseline simple. C'est un résultat
négatif propre, multi-seeds, à budget égal — exactement le genre qui mérite d'être écrit.

Fichiers : `listops_data.py`, `listops_model.py`, `listops_run.py`, `ood_curves.svg`,
checkpoint brut dans le scratchpad (`full_results.jsonl`).

---

## Option B — la tâche *dure* (SM + MED), pour donner sa chance à la hiérarchie

Faiblesse du test ci-dessus : le juge OOD ne discriminait pas, car R0 **extrapolait déjà**
(in ≈ ood). On a donc cherché une variante où le vanille se dégrade *vraiment* en profondeur,
pour offrir à la hiérarchie le seul terrain où elle pourrait gagner.

**Lever :** restreindre aux opérateurs **durs** SM (somme mod 10) + MED (médiane) — pas MAX/MIN,
trivialement « pool-friendly » — et tester jusqu'à profondeur 9. Calibration R0 seul : in 0,71 →
ood ~0,55. **Juge discriminant validé** (garde-fou 1+2 satisfaits) avant d'amener l'Etz.

**Résultats (3 seeds, budget égal, `ood_curves_hard.svg`) :**

| modèle | in (≤4) | ood5 | ood7 | ood9 |
|---|---|---|---|---|
| **R0** (vanille) | **0.737 ± 0.007** | 0.538 ± 0.015 | 0.557 ± 0.019 | 0.617 ± 0.016 |
| **Etz1** (1 monde) | 0.363 ± 0.007 | 0.305 ± 0.001 | 0.314 ± 0.002 | 0.326 ± 0.005 |
| **Etz4** (4 mondes + Tzimtzum) | 0.466 ± 0.024 | 0.387 ± 0.017 | 0.386 ± 0.016 | 0.397 ± 0.025 |

**Verdict : null FORT.**
1. **Etz4 ≫ Etz1 (+0.10)** encore — la profondeur aide dans la famille Etz (robuste sur les 2 variantes).
2. **Mais R0 écrase Etz4 (+0.27 in-dist)**, davantage que sur la tâche facile (+0.08). *Plus la tâche
   durcit, plus l'Etz décroche.* Même le pire de R0 (ood5 0.538) > le meilleur d'Etz4 (in 0.466).
3. **Piège à éviter :** la courbe d'Etz4 baisse *moins* en profondeur — mais c'est un **effet de
   plancher** (déjà bas, ne peut tomber de haut), pas de la robustesse. En absolu, R0 domine partout.

## Conclusion des deux options

- Ce qui survit : **les principes** (profondeur/hiérarchie aident quand on nourrit de structure).
- Ce qui est réfuté : **l'architecture** Etz n'est pas une bonne cellule de calcul — et c'est de
  plus en plus net à mesure que la tâche durcit, y compris sur un terrain *taillé pour elle*.
- La Voie B (vrais graphes) n'est **pas** justifiée : avant d'investir RDKit & co, il faudrait
  d'abord une cellule qui batte un baseline simple, ce qui n'arrive sur aucune des deux variantes.

---

## Phase 5 — Autopsie : QUI tue l'Etz ? (ablations diagnostiques, variante dure)

Plutôt que d'en rester à « l'Etz perd », on retire/modifie un composant à la fois pour localiser
la défaillance. Toutes les ablations gardent le comportement par défaut intact (switches optionnels).
Graphe : `ablation_autopsy.svg`. Résultats bruts : `results_ablation.jsonl`, `results_followup.jsonl`.

| modèle | params | in-dist | ood9 | rôle |
|---|---|---|---|---|
| Etz4 hyper | 1.11M | 0.466 ± 0.024 | 0.397 | le perdant d'origine |
| Etz4 **gated** (Tzimtzum porte) | 1.11M | 0.504 ± 0.055 | 0.425 | ablation Tzimtzum |
| Etz4 **direct** (kill-hypernet) | **205k** | **0.752 ± 0.069** | 0.629 | ablation hypernet |
| R0 plat (budget égal) | 208k | **0.780 ± 0.008** | 0.644 | baseline à budget égal |
| R0 plat | 1.13M | 0.737 ± 0.007 | 0.617 | baseline d'origine |
| Etz1 direct (1 monde) | 47k | 0.359 ± 0.005 | 0.329 | 1 monde, direct |

**Trois conclusions mécanistes :**

1. **Le hypernetwork EST le coupant n°1.** Le remplacer par des poids directs fait passer l'in-dist
   de 0.466 → 0.752 (**+0.29**), avec **5× moins de paramètres**. Le « partage massif de poids par
   génération », présenté comme l'innovation du projet, bridait sévèrement la capacité effective.

2. **Le Tzimtzum n'est qu'un coupant mineur** (+0.04 quand on le passe en porte apprise). Cohérent
   avec l'ablation Etz1-sans-Tzimtzum (0.36) < Etz4-avec (0.47) : le bottleneck était net-positif.

3. **La harm du hypernet s'amplifie avec la profondeur.** À 1 monde, hyper (0.363) ≈ direct (0.359) :
   aucune différence. À 4 mondes, direct (0.752) ≫ hyper (0.466). Le générateur partagé devient un
   goulot quand il doit produire 40 matrices distinctes (4 mondes × 10) au lieu de 10. *Leçon
   généralisable sur les hypernetworks.*

**Mais — le test décisif (à budget égal) ferme la porte au « win d'archi » :** à ~205k params,
R0 plat (0.780) **≥** Etz4 direct (0.752). Une fois le hypernet enlevé, la structure kabbalistique
(10 nœuds, 4 mondes, Tzimtzum) n'est **ni meilleure ni pire** qu'un simple DeepSets — c'est un
détour élaboré vers la même performance.

## Verdict final du projet

- **Le hypernet, pas la cosmologie, était la cause de l'échec** sur les terrains structurés.
- **Une fois corrigé, la structure n'apporte aucun avantage** sur un MLP plat à budget égal.
- Donc : pas de découverte d'architecture, mais un **résultat négatif mécaniste fort et propre**,
  avec une leçon réutilisable (les hypernetworks à génération peuvent coûter en capacité, d'autant
  plus que le nombre de jeux de poids demandés croît). C'est la vraie contribution.

---

## Phase 6 — « Quelle archi la tâche réclame-t-elle ? » — et le test de notre propre médecine

Raisonnement : les valeurs sont discrètes (0–9), les opérateurs sont symétriques → l'archi *optimale
en théorie* devrait être une cellule récursive qui (1) projette chaque enfant vers une valeur, (2)
**somme = histogramme** des valeurs, (3) lit le résultat via une tête conditionnée par l'opérateur.
MAX/MIN/MED/SM sont alors des fonctions *exactes* de l'histogramme → calcul exact à toute profondeur.
**Prédiction (écrite avant le run) : battre R0 (0,78) ET tuer la dégradation OOD.**

| modèle | params | in-dist | ood9 |
|---|---|---|---|
| R0 plat (mean-pool MLP) | 208k | **0,780** | 0,644 |
| **Hist** (histogramme + tête par op) | 104k | 0,463 ± 0,018 | 0,401 |
| **Hist_big** (budget R0) | 218k | 0,458 ± 0,014 | 0,392 |

**Prédiction RÉFUTÉE.** La cellule histogramme plafonne à ~0,46 (dès l'itération 1000, donc convergée,
pas sous-entraînée) — au niveau de l'Etz d'origine, loin sous le simple MLP.

**Pourquoi (diagnostic) :** la théorie était juste, mais les dynamiques d'apprentissage non. L'histogramme
à 10 cases via softmax est un **bottleneck dur et lossy**, *pire* que le mean-pool soft à 64 dim de R0 :
on donne *moins* d'info à la tête de lecture. Et le round-trip (valeur → vecteur 64d → re-softmax chez
le parent) doit s'aligner parfaitement et s'apprend mal ; les erreurs se composent.

**La leçon (la plus forte du projet) :** en concevant une archi structurée « théoriquement optimale »,
on est **tombé dans le piège même qu'on diagnostiquait** — elle perd contre le MLP plat, exactement
comme la cellule kabbalistique. C'est la **3ᵉ démonstration indépendante** de la thèse (cosmologie,
puis hypernet, puis notre propre histogramme), ce qui écarte tout soupçon de biais anti-Kabbale.

## Conclusion définitive

Sur ce calcul récursif, **les priors structurés rigides perdent systématiquement contre un MLP
générique** (mean-pool DeepSets), souple et haute-dimension, gardé compact. L'archi la plus performante
qu'on ait trouvée reste **R0** — précisément celle qui n'impose *aucune* structure maligne. Graphes :
`ablation_autopsy.svg`. Données : `results_histogram.jsonl`.

---

## Phase 7 — La première archi qui bat R0 : pooling multi-statistiques

Leçon des phases précédentes : ne pas *imposer une structure rigide*, mais *enrichir la cellule
générique de façon souple*. Candidat (proposé sans pari, après l'échec de l'histogramme) : remplacer
le seul `mean(enfants)` par **`[mean ; max ; min]`** concaténés → même MLP. Raison ancrée dans les
données : `max`/`min` fournissent MAX/MIN gratuitement (statistiques d'ordre que le mean-pool n'extrait
pas), l'info distributionnelle aide MED/SM, **en restant soft et haute-dim** (pas de bottleneck dur,
contrairement à l'histogramme). Données : `results_multistat.jsonl`.

| modèle | params | in-dist | ood5 | ood7 | ood9 |
|---|---|---|---|---|---|
| **MultiStat** [mean;max;min] | 207k | **0,791 ± 0,003** | **0,605** | **0,614** | **0,670** |
| R0 (mean seul) | 208k | 0,780 ± 0,008 | 0,577 | 0,601 | 0,644 |

**MultiStat bat R0** — modestement mais proprement : **+0,011 in-dist (3/3 seeds)**, **+0,026 ood9**
(2/3 seeds + meilleure moyenne + variance plus basse). C'est la **première** archi du projet à dépasser
le MLP plat. Gain réel mais petit : la tâche plafonne vers ~0,79 pour cette famille.

**Ce qui distingue ce succès des 3 échecs précédents :** ce n'est *pas* une structure imposée, c'est un
enrichissement **générique et soft** de l'agrégation. Cohérent de bout en bout : sur cette tâche, on
améliore en donnant plus de *features soft haute-dim* au MLP, jamais en imposant une géométrie rigide.

## Réponse finale : « quelle architecture est la plus performante ? »

**Une cellule récursive sur l'arbre, à pooling multi-statistiques soft `[mean ; max ; min]` + MLP,
gardée compacte (~200k).** Pas de cosmologie, pas de hypernet, pas de bottleneck, pas de discrétisation.
La leçon tient : *soft + générique + haute-dim* gagne ; *structuré + rigide* perd.

---

## Phase 8 — Sweep nocturne : le « plafond 0,79 » était du SOUS-ENTRAÎNEMENT (correction)

Sur conseil externe (« vérifie d'abord que la cellule est le goulot »), on a relancé R0 et les variantes
à **10 000 itérations (≈3,3×) + cosine schedule + 50 000 exemples** (vs 3000 / 15k), à budget égal,
variante dure. Données : `results_overnight.jsonl`. Graphe : `training_budget.svg`.

| modèle | in-dist | ood5 | ood9 | vs @3000 |
|---|---|---|---|---|
| R0 | 0.951 ± 0.004 | 0.853 | 0.884 | 0.780 → **+0.171** |
| MultiStat | 0.959 ± 0.003 | 0.892 | 0.911 | 0.791 → +0.168 |
| R0_resid | 0.949 ± 0.005 | 0.852 | 0.885 | — |
| **MultiStat_resid** | **0.962 ± 0.002** | 0.888 | 0.909 | — |
| Linear | 0.378 ± 0.001 | 0.310 | 0.335 | s'effondre |
| GRC | 0.456 ± 0.005 | 0.387 | 0.391 | ≈ @3000 (0.45) |

*(3 seeds pour R0/MultiStat/±résidual ; 2 seeds pour Linear/GRC — sweep arrêté à 16/18 par la fin de
fenêtre, les 2 runs manquants étant redondants. Les écarts vs l'agrégat 2-seeds sont <0,01.)*

**Cinq conclusions :**

1. **Le budget d'entraînement était LE goulot** — pas la cellule, pas la tâche. R0 passe de 0,78 à
   **0,954** (+0,17). **Cela corrige la conclusion « la tâche plafonne ~0,79 » des Phases 6-7** : c'était
   un artefact de sous-entraînement (3000 iters). L'advisor avait raison de l'imposer en priorité n°1.
2. **MultiStat ≥ R0 tient au budget élevé** (0,957 vs 0,954 ; ood9 0,912 vs 0,887 — net en OOD). Le gain
   du pooling [mean;max;min] est robuste, pas un artefact de petit budget.
3. **Résidual (#3) : ~neutre** (R0_resid ≈ R0 ; MultiStat_resid 0,960 ≈ MultiStat, peut-être +0,003 ood9).
   Le compounding n'était pas le levier ici.
4. **Linéaire (#2) : s'effondre** (0,378). La non-linéarité de composition est **nécessaire** (MED et le
   `mod` ne sont pas linéaires) — la théorie length-gen ne se traduit pas en « cellule littéralement linéaire ».
5. **GRC reste ~0,45 même à 10k** (≈ son score @3000) → son déficit est **intrinsèque à la cellule**
   (fold binaire séquentiel ≠ opérateurs symétriques K-aires), **pas** du sous-entraînement. Important :
   ça montre que le déficit des cellules structurées n'est *pas* un artefact de budget.

## Verdict révisé du projet

- La tâche **ne plafonnait pas** à 0,79 : c'était l'entraînement. Bien entraîné, le MLP atteint **~0,96**.
- **MultiStat (pooling soft enrichi) reste le meilleur**, surtout en extrapolation OOD.
- Les architectures **structurées rigides** (cosmologie, hypernet, histogramme, fold GRC) **restent
  derrière même à budget d'entraînement élevé** (GRC le confirme directement).
- *Limite ouverte :* on n'a pas re-testé Etz4/histogramme à 10k ; GRC@10k sert de témoin que leur
  déficit est intrinsèque, pas un manque d'entraînement.

---

## Phase 9 — L'invention qui MARCHE : l'agrégateur appris (Set-Attention / PMA)

Frontière B : MultiStat gagnait avec 3 statistiques *fixées à la main* (`mean;max;min`). Et si le
modèle **apprenait** quelles statistiques extraire ? La cellule `SetAggCell` utilise H **requêtes
apprises** qui attendent (softmax) sur les enfants → H résumés soft, invariants par permutation
(façon PMA / Set Transformer). Généralise MultiStat (mean = attention uniforme, max ≈ attention
piquée). Testée à budget égal, 10k+cosine+50k, variante dure. Données : `results_saa.jsonl`.
Graphe : `learned_aggregator.svg`.

| modèle | params | in-dist | ood5 | ood9 |
|---|---|---|---|---|
| **SAA_h4** (4 têtes apprises) | ~207k | **0.988 ± 0.002** | 0.968 | **0.971 ± 0.001** |
| SAA_h8 | ~207k | 0.986 ± 0.004 | 0.968 | 0.968 |
| SAA_h2 | ~207k | 0.986 ± 0.001 | 0.970 | 0.966 |
| SAA_hybrid (appris + max/min) | ~207k | 0.972 ± 0.003 | 0.944 | 0.942 |
| MultiStat (stats fixes) | ~207k | 0.959 | 0.891 | 0.911 |
| R0 (moyenne) | ~207k | 0.951 | 0.865 | 0.884 |

*(SAA_h2/h4 : 3 seeds ; SAA_h8/hybrid : 2 seeds au moment du dépouillement.)*

**Résultat : l'agrégation apprise bat nettement les statistiques fixes.** SAA_h4 atteint **0,988**
in-dist (vs 0,959 pour MultiStat) et surtout **0,971 en OOD profondeur 9 (vs 0,911)** — +0,06 en
extrapolation, la métrique qui compte. Robuste (3 seeds, ±0,002). **C'est la première contribution
architecturale *positive* du projet.**

Trois sous-résultats :
1. **~4 têtes suffisent** (h4 ≈ h8 > h2) : au-delà, aucun gain.
2. **L'hybride est PIRE** (0,972 < 0,988) : ajouter max/min « à la main » *gêne* une fois l'attention
   apprise en place. *Le pur appris bat l'appris + bricolé* — cohérent avec la thèse « pas de
   composant rigide superflu ».
3. **Le gain est surtout compositionnel** (OOD) : la cellule apprise généralise à des arbres bien
   plus profonds que ceux vus, signe qu'elle a appris une *vraie* règle de composition.

**Nuance honnête :** toujours sur notre variante dure, arbre gold donné, petit budget. Ce n'est pas
une comparaison au SOTA « arbre latent » (qui résout un problème différent). Mais dans notre
comparaison contrôlée, **SAA est le meilleur, de loin — et c'est une invention, pas un import.**

## Où en est l'architecture (résumé exécutif)

Le projet a maintenant **les deux moitiés** d'une bonne histoire de recherche :
- **La moitié négative** (Phases 1-6) : les biais structurés rigides (cosmologie, hypernet, histogramme)
  perdent contre un MLP générique ; le hypernet en est le coupant mécanique.
- **La moitié positive** (Phases 7-9) : on progresse en *enrichissant softement* l'agrégation — d'abord
  des statistiques fixes (MultiStat, +petit), puis un **agrégateur appris par attention (SAA, +net)**,
  qui améliore surtout la **généralisation compositionnelle**.

Prochaine frontière possible (A) : apprendre aussi la *structure* de l'arbre de façon soft, en
réutilisant SAA comme cellule.

---

## Tableau unifié définitif (consolidation)

Toutes les cellules au **même régime** (variante dure SM+MED, arbre gold, budget ~207k, 10k iters +
cosine + 50k données, 3 seeds sauf indication). C'est la comparaison de référence du projet.

| rang | cellule | in-dist | ood5 | ood9 | famille |
|---|---|---|---|---|---|
| 1 | **SAA_h4** (agrégat appris, 4 têtes) | **0.988** | 0.968 | **0.971** | soft appris |
| 2 | SAA_h2 | 0.986 | 0.970 | 0.966 | soft appris |
| 3 | SAA_h8 | 0.985 | 0.969 | 0.969 | soft appris |
| 4 | SAA_hybrid (appris + max/min) | 0.970 | 0.937 | 0.939 | soft appris |
| 5 | MultiStat_resid | 0.962 | 0.888 | 0.909 | soft fixe |
| 6 | MultiStat ([mean;max;min]) | 0.959 | 0.892 | 0.911 | soft fixe |
| 7 | R0 (mean-pool) | 0.951 | 0.853 | 0.884 | soft fixe |
| 8 | R0_resid | 0.949 | 0.852 | 0.885 | soft fixe |
| 9 | GRC (fold binaire gated) | 0.456 | 0.387 | 0.391 | structuré |
| 10 | Linear (composition linéaire) | 0.378 | 0.310 | 0.335 | dégénéré |
| 9b | Hist_big (histogramme) @10k | 0.517 | 0.431 | 0.457 | structuré |
| 9c | Etz4 (kabbalistique, hypernet) @10k | 0.469 | 0.395 | 0.415 | structuré |

**Lecture :** trois strates nettes. (1) **Agrégation apprise** (SAA) au sommet (~0,97-0,99). (2) **Agrégation
soft fixe** (MultiStat, R0) au milieu (~0,95-0,96). (3) **Cellules structurées/rigides** (GRC, linéaire)
tout en bas (~0,4). La progression *mean → stats fixes → attention apprise* monte à chaque marche, le
saut décisif étant « appris ». Les runs de consolidation (Etz4/Hist @10k) doivent confirmer que les
cellules structurées restent en bas **même bien entraînées** (GRC@10k le montre déjà).

---

## Phase 10 — Frontière A (apprendre la structure soft) + consolidation

### A) Consolidation : structuré reste bas même bien entraîné — ✅ confirmé

| modèle @10k+cosine | in-dist @3000 | in-dist @10k |
|---|---|---|
| Etz4 (kabbalistique, hypernet, 1,1M) | 0,466 | **0,469** (inchangé) |
| Hist_big (histogramme, ~208k) | ~0,46 | **0,517** (à peine bougé) |

Avec 3,3× d'entraînement, les cellules structurées restent à ~0,47-0,52, très loin du MLP (0,95) et
de SAA (0,99). **Leur déficit est intrinsèque à la cellule, pas un manque de budget** — la moitié
négative du projet est désormais inattaquable (Etz4, Hist ET GRC le confirment tous à 10k).

### B) Frontière A : apprendre la STRUCTURE de façon soft — ❌ négatif (pré-enregistré)

On donne au modèle une **séquence plate** (plus d'arbre gold) et il doit découvrir la composition,
via une fusion easy-first straight-through différentiable (`listops_parser.py`).

| modèle | in-dist | ood9 |
|---|---|---|
| borne haute (SAA, arbre gold) | 0,988 | 0,971 |
| **soft** (parseur appris) — 3 seeds | 0,469 | 0,321 |
| l2r (borne basse, sans structure) | 0,420 | 0,253 |

**Le parseur soft (0,469) est à peine au-dessus de la borne basse (0,420) et très loin du gold (0,988).**
Il apprend un peu de structure (+0,05) mais pas assez pour composer. C'est exactement l'issue
« soft ≈ l2r » pré-enregistrée : *l'induction de structure différentiable est instable ; battre le
beam search en pur-soft ne se fait pas ici.* Négatif honnête sur un vrai problème ouvert.

**Piste future (non tentée) :** Gumbel-softmax + recuit de température (exploration) pourrait aider —
c'est ce que la littérature utilise, notre ST-softmax simple a probablement collapsé vers une structure
quasi-triviale.

### Essai #2 — Gumbel + recuit de température : échec aussi (Frontière A fermée)

On a retenté avec la recette standard : bruit ST-Gumbel (exploration, train seulement) + recuit
linéaire τ 2,0 → 0,5. Données : `results_gumbel.jsonl`.

| modèle | in-dist | ood9 |
|---|---|---|
| gold (SAA) | 0,988 | 0,971 |
| soft (essai #1) | 0,469 | 0,321 |
| **gumbel (essai #2)** | **0,436 ± 0,006** | 0,311 |
| l2r (borne basse) | 0,420 | 0,253 |

Le Gumbel fait même légèrement *moins bien* que le soft glouton, et **stagne dès l'itération 2500**
(plateau total malgré le recuit), de façon identique sur 3 seeds. **Frontière A est fermée par deux
mécanismes indépendants** : ni le ST glouton ni Gumbel+recuit n'induisent la structure ici. Diagnostic
probable : le signal « classification finale » (1 label) est trop pauvre pour guider ~O(100) décisions
de fusion discrètes par exemple — c'est pourquoi le SOTA emploie une recherche semi-discrète (beam)
plutôt que du gradient pur. Problème ouvert, honnêtement cartographié.

### Essai #3 — supervision de structure (teacher forcing) : le diagnostic FINAL

Pour tester le diagnostic « signal trop pauvre », on a donné au parseur la **structure gold en
supervision** (teacher forcing des fusions à l'entraînement, CE sur le scorer), avec deux poids de
perte de structure, et une éval double : parsing **libre** (in/ood) et **teacher-forcé** (`in_tf` =
plafond du composeur, structure parfaite donnée). Au passage, correction d'un bug critique (les
séquences courtes d'un batch fusionnaient leur racine avec le padding). Données :
`results_sup_w02.jsonl`, `results_sup_w10.jsonl`.

| config (3 seeds) | in (libre) | in_tf (structure gold) | ood9 |
|---|---|---|---|
| sup, struct_w=0.2 | 0.258 ± 0.026 | 0.365 ± 0.035 | 0.180 |
| sup, struct_w=1.0 | **0.411 ± 0.003** | **0.454 ± 0.001** | 0.157 |
| rappel : l2r (fold fixe) | 0.420 | — | 0.253 |
| rappel : GRC gold-tree @10k | 0.456 | (structure gold par construction) | 0.391 |
| rappel : SAA gold-tree (K-aire) | — | 0.988 | 0.971 |

**Le verdict, et il est élégant : le goulot n'a JAMAIS été la découverte de structure.** Avec la
supervision pleine (w=1.0), le parsing *libre* (0.411) rejoint quasiment le plafond teacher-forcé
(0.454) — la structure est donc bien apprise/suivie. Mais ce plafond lui-même (~0.45) est **identique
au GRC gold-tree (0.456) et à peine au-dessus du fold fixe l2r (0.420)**. Autrement dit : *toutes* les
approches par **composition binaire séquentielle** — l2r, GRC, soft, gumbel, supervisée — butent sur
le **même mur ~0.42-0.46**, quelle que soit la manière dont la structure est choisie.

**Diagnostic mécaniste final de Frontière A :** ce n'est pas « apprendre l'arbre » qui bloque, c'est
la **binarisation elle-même**. Nos opérateurs durs sont K-aires et symétriques ; MED, en particulier,
n'est **pas décomposable par paires** (la médiane de médianes est fausse) — un fold binaire doit donc
transporter tout le multiset dans son état, ce qui ne s'apprend pas dans notre régime. C'est
exactement pourquoi la SAA (agrégation **K-aire** en un coup) atteint 0.988 là où toute composition
binaire plafonne à ~0.45. L'échec du GRC (Phase 8) et celui du parseur (essais 1-3) sont **le même
échec**, enfin unifié.

## Phase 11 — Le SAA-Parser : de la séquence plate à ~0,91, la synthèse des deux frontières

Le diagnostic de la Phase 10 (« la binarisation est le mur, pas la structure ») **désigne** une
architecture non explorée : un parseur à structure latente qui fusionne des **groupes K-aires entiers**
(opérateur + span d'opérandes, fenêtre K+2 fixe) avec la **SAA comme composeur**, au lieu de paires.
Conception écrite avant toute expérience : `SAA_PARSER_PLAN.md`. Implémentation : `listops_saa_parser.py`
(avec test unitaire de padding exécuté avant tout run — il a immédiatement attrapé un vrai bug : à
fenêtre unique, la « dernière fenêtre » du padding coïncidait avec la racine ; fixé).

*Note de rigueur (budget-matching) :* le premier run du seed 0 de `sup` avait accidentellement hérité
d'un budget résiduel (486 986 params, hidden=512, d'un essai avorté) au lieu de 191 562 (hidden=256,
comme les seeds 1-2). Détecté et corrigé (seed 0 relancé au bon budget) avant de figer les chiffres
ci-dessous — les 3 seeds de `sup` sont maintenant strictement budget-matchés à `free` et à l'ablation.

### 11.1 Étape 1 — validation causale du diagnostic + premier test œuf-poule (3 seeds)

| mode | in-dist | ood5 | ood9 |
|---|---|---|---|
| **sup** (structure gold donnée, teacher forcing) | **0,925 ± 0,009** | 0,670 ± 0,019 | 0,435 ± 0,056 |
| **free** (structure apprise from scratch) | 0,660 ± 0,021 | 0,329 ± 0,011 | 0,278 ± 0,008 |
| *rappel* : mur binaire (toute variante, Phase 10) | ~0,42–0,46 | — | ~0,25–0,39 |
| *rappel* : SAA sur arbre gold (Phase 9) | 0,988 | — | 0,971 |

Le passage binaire → K-aire (avec structure donnée) fait **+0,46 à +0,50** — le diagnostic
« binarisation = coupable » est **prouvé causalement**, très au-delà du seuil ≥0,9 pré-enregistré.
En parsing libre, le K-aire bat déjà largement le mur binaire (0,660 vs 0,45) : hypothèse œuf-poule
**partiellement** validée à ce stade — la structure aide, mais l'écart au plafond sup reste large (0,265).

### 11.2 Ablation : la fenêtre K-aire ou le composeur SAA — qui porte le gain en mode libre ?

Même parseur, cellule interchangeable (`--cell`), mode free, 3 seeds :

| composeur (mode free) | in-dist | ood9 |
|---|---|---|
| vanilla (mean-pool) | 0,634 ± 0,018 | 0,269 ± 0,011 |
| multistat ([mean;max;min]) | 0,628 ± 0,003 | 0,284 ± 0,009 |
| **SAA** (attention apprise) | 0,660 ± 0,021 | 0,278 ± 0,008 |

**Résultat inattendu et important à documenter honnêtement :** en mode libre, les trois composeurs
sont **quasi indiscernables** (~0,63–0,66). Contrairement au régime supervisé (où SAA écrase les
cellules binaires, 0,93 vs 0,46), **la qualité du composeur ne se manifeste pas quand la structure
doit être découverte** — les erreurs de parsing dominent et noient l'avantage du composeur. Le gros
saut binaire→K-aire (0,45→0,63) vient donc principalement de la **fenêtre K-aire elle-même**
(regrouper K opérandes d'un coup), pas spécifiquement de l'attention apprise. La SAA ne « gagne son
loyer » qu'une fois la structure correcte acquise.

### 11.3 Curriculum : la validation forte de l'hypothèse œuf-poule

Test direct, conçu cette nuit : geler la structure en teacher forcing plein pendant les 30% premiers
pas (`curr_hold=0.3`), puis sevrage **linéaire** vers le parsing libre. Intuition : laisser le
composeur devenir compétent *avant* de lui confier la découverte de structure — au lieu des deux
d'un coup (mode `free`).

| mode | in-dist | ood5 | ood9 |
|---|---|---|---|
| free (from scratch) | 0,660 ± 0,021 | 0,329 ± 0,011 | 0,278 ± 0,008 |
| **curriculum** (hold 30% puis sevrage) | **0,909 ± 0,017** | 0,647 ± 0,032 | 0,459 ± 0,041 |
| sup (plafond, structure toujours donnée) | 0,925 ± 0,009 | 0,670 ± 0,019 | 0,435 ± 0,056 |

**Le curriculum comble presque tout l'écart** : 0,909 contre un plafond de 0,925 (**98,3 %** du plafond),
très loin au-dessus du free-from-scratch (0,660) et du mur binaire (~0,45), robuste sur 3 seeds
(σ=0,017). **L'hypothèse œuf-poule est fortement validée** : l'induction de structure non supervisée
échouait bien parce que le composeur ne savait rien calculer au début (aucun signal pour distinguer
une bonne d'une mauvaise fusion) ; lui laisser le temps d'apprendre à calculer *avant* de lâcher la
structure débloque presque tout le potentiel. Bonus : le curriculum améliore aussi nettement l'OOD
(ood9 0,459 vs 0,278 pour le free) — meilleure généralisation compositionnelle, pas seulement
meilleur score in-dist.

### 11.4 Généralité : les 4 opérateurs (ListOps complet, pas seulement SM+MED)

| mode | in-dist | ood9 |
|---|---|---|
| free, MAX+MIN+MED+SM (4 ops) | **0,753 ± 0,007** | 0,473 ± 0,004 |
| *rappel* : free, SM+MED seuls (variante dure) | 0,660 ± 0,021 | 0,278 ± 0,008 |

Sans surprise (MAX/MIN sont plus faciles, cf. Phases 4–4bis), le SAA-Parser en mode libre fait
nettement mieux sur ListOps complet que sur la variante durcie — le mécanisme généralise au-delà du
sous-ensemble d'opérateurs difficiles utilisé pour le diagnostic.

### 11.5 Limite qui persiste : la dégradation en profondeur (OOD)

Même les meilleurs modes (sup 0,925→0,435 ; curriculum 0,909→0,459) perdent ~50% de leur accuracy
entre in-dist et profondeur 9. Le curriculum et la supervision résolvent « structure + calcul », pas
la généralisation compositionnelle profonde — un problème distinct, non résolu par ce travail.

## Bilan final des deux frontières d'invention

- **Frontière B (agrégation apprise, SAA sur arbre gold)** : ✅ réussie — 0,988.
- **Frontière A (structure apprise depuis une séquence plate)** : ✅ **réussie aussi, via le
  SAA-Parser + curriculum** — 0,909, à 98,3% du plafond supervisé, très loin au-dessus de toute
  composition binaire (~0,45) et du parsing libre from-scratch (0,660).

**Le résultat-phare du projet** : un parseur à structure latente K-aire, entraîné avec un curriculum
qui laisse le composeur devenir compétent avant de lui confier la structure, résout le problème que
la littérature attaque avec du beam search discret — ici, entièrement par gradient, sans recherche.
C'est la synthèse complète des deux frontières, désignée par le diagnostic mécaniste de la Phase 10,
conçue et validée selon un protocole pré-enregistré. Nuance honnête conservée : (a) l'ablation montre
que le gain vient d'abord de la fenêtre K-aire, la SAA n'ajoutant sa valeur qu'une fois la structure
acquise ; (b) la généralisation en profondeur (OOD) reste un problème ouvert, non résolu ici.

---

## Phase 12 — Généralisation #1 : arité variable [2,7] (résultat mitigé, documenté honnêtement)

Plan pré-enregistré : `VARIABLE_ARITY_PLAN.md`. Arité vraiment variable par nœud, fenêtre canonique
Kmax+2=9 avec token PAD masqué explicitement dans le composeur. Données : `results_vararity.jsonl`.

| mode (3 seeds, hidden=192) | in | ood9 | in_tf |
|---|---|---|---|
| sup | 0,611 ± 0,008 | 0,347 | 0,611 |
| curr | 0,613 ± 0,003 | 0,366 | 0,613 |
| *rappel K=5 fixe (Phase 11)* | *0,925 / 0,909* | *0,435 / 0,459* | — |

Selon la grille pré-enregistrée : **sup < 0,7 → le régime padding/arité variable introduit une
difficulté propre**. Le plafond chute à ~0,61 *même avec structure gold* (in_tf = in) — ce n'est donc
pas la découverte de structure qui casse, mais le *calcul* sous arité variable. Point positif : le
curriculum atteint 100% de son propre plafond (0,613 ≈ 0,611) — **le mécanisme du curriculum
généralise**, c'est le plafond du composeur qui baisse. Caveats : hidden=192 (vs 256 en Phase 11) ;
et surtout, voir Phase 13 — les séquences paddées étant plus longues, une partie du collapse pourrait
être *positionnelle* (follow-up `results_vararity_nopos.jsonl` en cours).

## Phase 13 — Généralisation #2 : SANS embeddings positionnels — LE PROBLÈME OUVERT EST RÉSOLU

Hypothèse pré-enregistrée (NIGHT3_PLAN.md) : la chute OOD en profondeur (~50% perdus, « problème
ouvert » des Phases 11) vient des embeddings positionnels absolus, sous-entraînés aux positions
longues (les séquences OOD sont plus longues) ; le scan de fenêtres est *local* et n'en a pas besoin.
Config identique Phase 11 (hidden 256, SM+MED, K=5 fixe), seul `--no_pos` change.
Données : `results_nopos.jsonl`.

| mode (3 seeds) | in | ood5 | ood9 |
|---|---|---|---|
| sup avec pos (Phase 11) | 0,925 | 0,670 | 0,435 |
| **sup SANS pos** | **0,968 ± 0,002** | **0,933** | **0,930 ± 0,013** |
| curr avec pos (Phase 11) | 0,909 | 0,647 | 0,459 |
| **curr SANS pos** | **0,965 ± 0,003** | **0,915** | **0,917 ± 0,012** |

**C'est le plus gros résultat unitaire du projet.** Retirer les positions : (a) améliore même
l'in-dist (+0,04) ; (b) fait passer l'OOD profondeur 9 de **0,46 à 0,92** — la dégradation en
profondeur disparaît presque entièrement (0,968 → 0,930, −0,04 seulement). La généralisation
compositionnelle profonde, listée « problème distinct non résolu » en Phase 11, est **résolue** par
une soustraction : le parseur K-aire est intrinsèquement local, les positions absolues ne faisaient
qu'empoisonner l'extrapolation. Cohérent avec la thèse du projet : *encore un composant superflu
qui nuisait*.

## Phase 14 — Généralisation #3 : opérateurs jamais vus (MODE, RNG)

Sémantiques d'une autre nature (MODE = plus fréquent → comptage ; RNG = max−min → combinaison de
statistiques d'ordre), ids d'opérateurs neufs, harnais identique (K=5, hidden 256, avec positions).
Données : `results_newops.jsonl`.

| mode (3 seeds) | in | ood5 | ood9 |
|---|---|---|---|
| sup | 0,993 ± 0,001 | 0,947 | 0,811 |
| curr | 0,995 ± 0,003 | 0,947 | 0,852 |

**Quasi parfait in-dist (0,995)** et OOD très solide — le mécanisme n'a rien de spécifique à SM/MED :
il transfère à des calculs de nature différente (et s'avère même *plus facile* sur MODE/RNG que sur
SM/MED). La généralisation à travers les sémantiques d'opérateurs est démontrée.

## Bilan de généralisation (nuit #3)

| axe testé | verdict |
|---|---|
| arité variable | ⚠️ plafond chute (~0,61) — difficulté propre au régime paddé ; curriculum toujours à 100% de son plafond ; follow-up no_pos en cours |
| profondeur OOD | ✅ **résolue** en retirant les positions (0,46 → 0,92 à prof. 9) |
| nouveaux opérateurs | ✅ transfère (0,995) |

---

## Nuit #4 (Phases 15-17) — la dernière limite levée + le chiffre-vitrine

Plan pré-enregistré : `NIGHT4_PLAN.md`. Données : `results_vararity_nopos.jsonl`,
`results_var_count.jsonl`, `results_capstone.jsonl`, `results_ch01/ch05.jsonl`.

### Phase 15 — La chute d'arité variable n'était PAS positionnelle… c'était le COMPTE

(a) **Contrôle no_pos** : arité variable sans positions = 0,615/0,613 (sup/curr) ≈ 0,611 avec
positions → la chute n'est pas un artefact positionnel (contrairement à la chute OOD, Phase 13).
Bonus cohérent : l'ood9 monte quand même (0,49 vs 0,35).

(b) **Diagnostic mécanique désigné par l'architecture** : l'attention masquée calcule des
*moyennes pondérées* ; or SM = somme exige le *nombre* d'opérandes. À K fixe, somme = K×moyenne
(constante apprenable) ; à K variable, le compte est détruit par le softmax. **Fix : un embedding
du compte d'enfants valides** réinjecté dans le composeur (`--count`, +13k params).

| arité variable [2,7] (3 seeds) | in | ood9 |
|---|---|---|
| avec positions (Phase 12) | 0,611 / 0,613 | 0,35 / 0,37 |
| no_pos seul | 0,615 / 0,613 | 0,50 / 0,49 |
| **no_pos + count** | **0,895 ± 0,004 / 0,905 ± 0,011** | **0,80 / 0,79** |

Selon la grille pré-enregistrée (sup ≥ 0,85) : **diagnostic PROUVÉ**. La limite d'arité variable
était le compte perdu — un embedding la lève (+0,29). Troisième fois dans le projet qu'un diagnostic
mécaniste désigne le fix exact (hypernet → poids directs ; OOD → no_pos ; arité → count).

### Phase 16 — Le chiffre-vitrine : UN modèle, SIX opérateurs, séquences plates

Tout ce qui a été appris, combiné : K=5, no_pos, curriculum, mélange MAX+MIN+MED+SM+MODE+RNG.

| mode (3 seeds) | in | ood5 | ood9 |
|---|---|---|---|
| sup | 0,929 ± 0,004 | 0,891 | 0,920 |
| **curriculum** | **0,933 ± 0,004** | 0,892 | **0,915 ± 0,002** |

**Un seul modèle de 192k params apprend, depuis des séquences plates sans arbre : la structure, le
calcul de 6 opérateurs de natures différentes, et généralise quasi à plat en profondeur**
(0,933 → 0,915 à profondeur 9, −0,018 seulement). Le curriculum égale voire dépasse le plafond
supervisé. C'est le résultat-vitrine du projet.

### Phase 17 — Le curriculum est robuste (pas un hyperparamètre fragile)

curr_hold = fraction d'entraînement en structure gelée avant sevrage (0,3 partout ailleurs) :

| curr_hold | in | ood9 |
|---|---|---|
| 0,1 | 0,965 ± 0,004 | 0,927 |
| 0,3 (réf. Phase 13) | 0,965 ± 0,003 | 0,917 |
| 0,5 | 0,969 ± 0,004 | 0,930 |

Identique sur toute la plage 0,1–0,5 : **c'est le sevrage progressif qui compte, pas le réglage du
hold**. Le mécanisme n'est pas fragile.

## État final du projet après la nuit #4

| limite documentée | statut |
|---|---|
| dégradation OOD en profondeur | ✅ levée (no_pos, Phase 13) |
| arité variable | ✅ levée (count-feature, Phase 15) |
| spécificité aux opérateurs | ✅ levée (MODE/RNG + capstone 6 ops, Phases 14/16) |
| sensibilité du curriculum | ✅ écartée (Phase 17) |
| échelle / domaines réels (AST, molécules) | ⏳ non testé |
| comparaison à un parseur latent publié, au même budget | ⏳ non faite |

Figure de synthèse des Phases 11 à 17 : `generalization_summary.svg`.

### Réserves sur ce bilan (à lire avec le tableau)

- **« Levée » veut dire « levée dans ce cadre ».** Tout est mesuré sur ListOps et ses variantes.
- **Pas de comparaison externe.** Les références binaires (l2r, GRC, soft, Gumbel, supervisé) sont nos
  implémentations. Aucun parseur latent de la littérature n'a été entraîné au même budget : on peut
  dire que le SAA-Parser fonctionne, pas qu'il bat l'existant.
- **Arité variable : levée à 0,90, pas à 0,965.** Il reste ~0,06 sous le régime à arité fixe, avec une
  arité bornée (Kmax=7) et des emplacements vides signalés par un token PAD. Ces runs ont un MLP plus
  étroit (hidden 192, 148–161k paramètres) que ceux à arité fixe (hidden 256, 192k).
- **Mode libre sans positions : non mesuré.** Les Phases 13 à 17 n'ont que `sup` et `curr`. Le 0,660
  du parsing libre (Phase 11) date de la configuration avec positions.
- **Le curriculum utilise l'arbre gold à l'entraînement.** Seul le mode `free` s'en passe.

---

## Index des fichiers de données (un run par ligne JSON)

| fichier | phase | contenu |
|---|---|---|
| `results_easy.jsonl` | 4 | R0, Etz1, Etz4 — ListOps 4 opérateurs, 3 seeds |
| `results_hard.jsonl` | 4-bis | R0, Etz1, Etz4 — variante dure SM+MED |
| `results_ablation.jsonl` | 5 | Etz4_direct (sans hypernet), Etz4_gated (Tzimtzum porte) |
| `results_followup.jsonl` | 5 | R0 à budget égal (~208k), Etz1_direct |
| `results_histogram.jsonl` | 6 | Hist, Hist_big (cellule histogramme) |
| `results_multistat.jsonl` | 7 | MultiStat à 3000 itérations |
| `results_overnight.jsonl` | 8 | R0, R0_resid, MultiStat, MultiStat_resid, Linear, GRC à 10k + cosine |
| `results_grc3k.jsonl` | 8 | GRC à 3000 itérations (témoin) |
| `results_saa.jsonl` | 9 | SAA_h2, SAA_h4, SAA_h8, SAA_hybrid sur arbre gold |
| `results_parser.jsonl` | 10, essai 1 | parseur binaire : l2r (fold fixe), soft (straight-through) |
| `results_gumbel.jsonl` | 10, essai 2 | parseur binaire Gumbel + recuit |
| `results_sup_w02.jsonl`, `results_sup_w10.jsonl` | 10, essai 3 | parseur binaire supervisé, poids de structure 0,2 et 1,0 |
| `results_consol.jsonl` | 10, consolidation | Etz4 et Hist_big ré-entraînés à 10k |
| `results_saaparser.jsonl` | 11.1 | SAA-Parser `sup` et `free` (seed 0 de `sup` relancé au bon budget) |
| `results_saap_vanilla.jsonl`, `results_saap_ms.jsonl` | 11.2 | ablation du composeur en mode `free` |
| `results_saap_curr.jsonl` | 11.3 | curriculum (hold 30 %) |
| `results_saap_allops.jsonl` | 11.4 | mode `free`, 4 opérateurs |
| `results_vararity.jsonl` | 12 | arité variable 2–7, avec positions |
| `results_nopos.jsonl` | 13 | arité fixe, sans positions |
| `results_newops.jsonl` | 14 | MODE + RNG, avec positions |
| `results_vararity_nopos.jsonl` | 15 | arité variable, sans positions |
| `results_var_count.jsonl` | 15 | arité variable, sans positions, avec compte |
| `results_capstone.jsonl` | 16 | 6 opérateurs, sans positions |
| `results_ch01.jsonl`, `results_ch05.jsonl` | 17 | curriculum avec hold 10 % et 50 % |

**Attention pour rejouer les Phases 4 à 12.** Depuis la Phase 13 (nuit #3), `listops_data.py` déclare six
opérateurs (MODE et RNG ajoutés en fin de liste). Les identifiants de MAX, MIN, MED, SM sont inchangés,
mais le vocabulaire a deux entrées de plus : un modèle ré-entraîné aujourd'hui avec `--ops SM,MED` a
quelques paramètres de plus que celui des fichiers ci-dessus (par exemple 191 690 au lieu de 191 562
pour le SAA-Parser) et une initialisation différente à seed égal. Les chiffres stockés restent valides ;
une reproduction au bit près des phases antérieures demande de revenir à la liste de quatre opérateurs.
