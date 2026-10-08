# L'Arbre de Vie en deep learning : d'un résultat négatif sur les biais inductifs kabbalistiques à un parseur latent K-aire entraîné par curriculum

*Projet Etz HaChaim AI — rapport court (workshop-style, ~5 pages).*

## Résumé

Peut-on traduire la cosmologie de la Kabbale (l'Arbre de Vie : 4 Mondes, 10 Sefirot, 22 chemins,
contraction « Tzimtzum ») en une architecture de réseau de neurones compétitive ? Nous
implémentons cette architecture via des *Fractal Hypernetworks* et la confrontons à des baselines
standard sur deux familles de tâches : modélisation de texte au caractère, puis évaluation
d'expressions arithmétiques imbriquées (ListOps). **Résultat principal (négatif) :** l'architecture
kabbalistique perd systématiquement contre un simple MLP, et l'écart se creuse à mesure que la tâche
durcit. **Diagnostic mécaniste (par ablations) :** le coupable n'est pas la cosmologie mais le
*hypernetwork générateur de poids* ; le remplacer par des poids directs récupère ~0,29 d'accuracy
avec 5× moins de paramètres, et sa nuisance croît avec la profondeur. **Première contribution
positive :** une fois la structure rigide abandonnée, on améliore la baseline en *enrichissant
softement* l'agrégation — culminant dans un **agrégateur appris par attention sur ensemble (SAA)**
qui porte l'accuracy à **0,988** (vs 0,959) sur arbre donné. **Diagnostic n°2, en essayant d'apprendre
la structure elle-même** (depuis une séquence plate, sans arbre gold) : trois méthodes de composition
*binaire* (glouton, Gumbel, supervisé) plafonnent toutes à ~0,45 — la découverte de structure n'était
pas le problème, **la binarisation l'était** (nos opérateurs sont K-aires et symétriques ; la médiane
n'est pas décomposable par paires). **Deuxième contribution positive, désignée par ce diagnostic :**
un **parseur à structure latente qui fusionne des groupes K-aires** avec la SAA comme composeur,
entraîné par un **curriculum** (geler la structure le temps que le composeur devienne compétent, puis
sevrage progressif), atteint **0,909 ± 0,017** depuis une séquence plate — 98,3% du plafond obtenu
avec la structure donnée (0,925), très loin au-dessus de toute composition binaire (~0,45) et du
parsing libre from-scratch sans curriculum (0,660). Une ablation montre une nuance importante : en
mode libre, la qualité du composeur (SAA vs mean-pool) ne se distingue presque pas — l'avantage de la
SAA n'émerge qu'une fois la structure correctement acquise. **Trois tests de généralisation** lèvent
ensuite les limites restantes, chaque correctif étant désigné par un diagnostic : (i) la dégradation
en profondeur était un artefact des embeddings positionnels absolus — les retirer fait passer
l'accuracy à profondeur 9 de 0,46 à **0,92** et améliore même l'in-distribution (0,965) ; (ii) sous
arité variable, le plafond tombe à 0,61 parce que l'attention calcule des moyennes et perd le *nombre*
d'opérandes — réinjecter ce compte le remonte à **0,90** ; (iii) le mécanisme transfère à des
opérateurs jamais vus (0,99). **Configuration finale :** un seul modèle de 192k paramètres, six
opérateurs mélangés, sans positions, entraîné par curriculum sur séquences plates, atteint
**0,933 in-distribution et 0,915 à profondeur 9**. Sur ce calcul récursif : **les priors structurés
rigides perdent contre des features soft et haute-dimension ; et une structure latente K-aire
s'apprend par curriculum, entièrement par gradient, sans recherche discrète.** Deux réserves cadrent
ces résultats : tout repose sur une tâche synthétique (ListOps), et nous n'avons pas comparé le
parseur à une méthode d'arbre latent publiée, au même budget.

## 1. Introduction

L'idée de départ est radicale : la Kabbale décrit la création comme une descente de l'information à
travers 4 Mondes hiérarchiques, chacun composé de 10 Sefirot (nœuds) reliées par 22 chemins, avec une
contraction (*Tzimtzum*) entre les niveaux. Ces motifs — hiérarchie d'abstraction, bottleneck,
modularité, partage de paramètres — ressemblent à de bons *biais inductifs*. La question scientifique :
ces motifs, encodés littéralement, produisent-ils une architecture compétitive, ou la belle métaphore
ne survit-elle pas au contact des données ?

Nos contributions :
1. un **résultat négatif propre** (multi-seeds, budget égal, prédictions pré-enregistrées) montrant que
   l'architecture kabbalistique est dominée par un MLP générique ;
2. une **localisation mécaniste** de la défaillance par ablations — le hypernetwork, pas la cosmologie ;
3. une **règle de conception positive** : sur ce type de tâche, on progresse par enrichissement soft,
   pas par structure rigide — y compris contre une architecture que nous pensions optimale ;
   culminant dans un agrégateur appris par attention sur ensemble (SAA, §4.6) ;
4. un **diagnostic causal** de l'échec de l'induction de structure (la binarisation, pas la découverte
   de structure, §4.7), qui **désigne** une architecture que nous n'avions pas essayée — un parseur à structure latente
   K-aire — validée par un **curriculum** atteignant 98,3% d'un plafond entièrement supervisé, en
   partant d'une séquence plate et sans recherche discrète (§4.8) ;
5. trois **tests de généralisation** (§4.9) dont deux aboutissent à un correctif désigné par
   diagnostic : retirer les embeddings positionnels supprime la dégradation en profondeur, et
   réinjecter le compte d'opérandes rétablit le calcul sous arité variable.

## 2. L'architecture Etz HaChaim et le « renversement » récursif

**Le cœur fractal.** L'information traverse `num_worlds` Mondes successifs. Chaque Monde est un graphe
de 10 nœuds (Sefirot) avec une matrice d'adjacence 10×10 *apprise* (les « 22 chemins » ne sont donc pas
fixes). Les poids de transformation de chaque nœud sont *générés* par un hypernetwork conditionné sur
la position (monde, sefirah) — un partage de paramètres massif. Entre deux Mondes, une couche Tzimtzum
compresse puis ré-étend la représentation (bottleneck). Des switches permettent les ablations : 1 Monde
sans Tzimtzum (E0/Etz1) vs 4 Mondes complets (E4/Etz4).

**Le défaut initial.** Dans la version « texte », l'entrée était copiée à l'identique sur les 10 Sefirot
(`x = x.repeat(1, 10, 1)`) : la structure de graphe n'était jamais alimentée par la donnée. L'Arbre
était, de fait, un MLP déguisé.

**Le renversement.** Pour faire entrer une *vraie* structure, nous transformons l'Etz en **cellule
récursive appliquée à chaque nœud d'un arbre de syntaxe** (ListOps). À chaque nœud interne :
l'opérateur entre dans la Sefirah S0, chaque opérande dans une Sefirah distincte (S1..S5), les Sefirot
restantes servent de registres de travail, et le résultat est lu sur la dernière Sefirah (Malchut). La
même cellule (poids partagés) est appliquée bottom-up sur tout l'arbre — la profondeur variable étant
absorbée par la récursion. C'est la première configuration où la structure entre réellement par la
donnée plutôt que par recopie.

## 3. Protocole expérimental

**Tâche.** ListOps : expressions imbriquées d'opérateurs (MAX, MIN, MED = médiane, SM = somme mod 10)
sur des chiffres 0–9 ; sortie = un chiffre (classification à 10 classes). Arité fixe K=5 ; profondeur
contrôlée ; taille des arbres plafonnée (isole la profondeur comme seule variable du test
d'extrapolation, et borne la mémoire). Pour les sections 4.1–4.6, on donne l'arbre gold à tous les
modèles : on teste donc la *cellule de combinaison*, pas la découverte de structure. Cette hypothèse
est explicitement levée en 4.7–4.9, où les modèles reçoivent une séquence plate sans arbre. La
section 4.9 ajoute deux opérateurs (MODE = valeur la plus fréquente, RNG = max − min) et une variante
à arité variable (2 à 7 opérandes par nœud).

**Garde-fous (fixés avant les runs).**
- *Calibration* : on ne lance les modèles Etz que si la baseline ne sature pas in-distribution.
- *Juge n°1 = extrapolation* : performance sur arbres plus profonds que ceux vus à l'entraînement.
- *Budget égal* : tous les modèles comparés à ~même nombre de paramètres.
- *3 seeds* (exceptions listées en §6), moyenne ± écart-type ; verdicts pré-enregistrés.

**Baseline (R0).** Cellule récursive « DeepSets » : pooling moyenne des enfants + embedding de
l'opérateur → MLP. Permutation-invariante (les opérateurs sont symétriques), souple, haute-dimension.

## 4. Résultats

### 4.1 Tournoi texte (Tiny Shakespeare, validation loss, plus bas = mieux)

| Bigram | Etz 4 mondes (1,1M) | Etz + attention | **Transformer (816k)** |
|---|---|---|---|
| 3,45 | 2,49 | 2,19 | **1,80** |

L'Arbre apprend (il écrase le Bigram) mais perd nettement contre le Transformer ; surtout, passer de
1 à 4 Mondes n'apporte rien (E0 ≈ E4). Diagnostic : un graphe spatial appliqué à une séquence, sans
structure d'entrée réelle.

### 4.2 ListOps : la structure entre par la donnée (accuracy, 3 seeds)

| variante | R0 (vanille) | Etz 1 monde | Etz 4 mondes |
|---|---|---|---|
| facile (4 ops) | **0,745** | 0,536 | 0,664 |
| dure (SM, MED) | **0,737** | 0,363 | 0,466 |

Deux faits : (a) dès que la structure entre, **la profondeur aide** (Etz4 ≫ Etz1) — la malédiction
E0≈E4 du texte est brisée ; (b) mais l'Etz reste **sous** le MLP plat, et l'écart se **creuse** sur la
tâche dure (−0,08 → −0,27).

### 4.3 Autopsie : localiser la défaillance (variante dure, budget égal)

| modèle | params | in-dist |
|---|---|---|
| Etz4 hyper | 1,1M | 0,466 |
| Etz4 + Tzimtzum *gated* | 1,1M | 0,504 |
| **Etz4 sans hypernet (poids directs)** | **205k** | **0,752** |
| R0 plat | 208k | 0,780 |

**Le hypernetwork est le coupable.** Le remplacer par des poids directs fait **+0,29** avec **5× moins
de paramètres**. La contraction Tzimtzum n'est qu'un facteur mineur (+0,04 en porte apprise). De plus,
la nuisance du hypernet **croît avec la profondeur** : à 1 Monde, poids générés ≈ poids directs (≈0,36) ;
à 4 Mondes, l'écart explose. Le générateur partagé devient un goulot quand il doit produire davantage
de jeux de poids distincts. Mais à budget égal, le MLP plat (0,780) **≥** l'Etz corrigé (0,752) : une
fois le hypernet enlevé, la structure cosmologique n'est ni meilleure ni pire qu'un simple DeepSets.

### 4.4 Quelle architecture la tâche réclame

Nous avons conçu une cellule *théoriquement optimale* (les valeurs sont discrètes, les opérateurs
symétriques) : projeter chaque enfant vers une valeur, **sommer = histogramme**, lire via une tête
conditionnée par l'opérateur — d'où un calcul exact à toute profondeur. **Prédiction écrite à l'avance :
battre R0.** Elle a échoué (≈0,46, niveau Etz d'origine) : le softmax 10-cases est un bottleneck dur et
lossy, pire que le mean-pool soft 64-dim. En revanche, un enrichissement *soft* de l'agrégation —
pooling multi-statistiques `[mean ; max ; min]` — **bat R0** : **0,791 vs 0,780** (3/3 seeds, budget
égal), avec un gain plus net en extrapolation. `max`/`min` fournissent MAX/MIN gratuitement tout en
restant soft et haute-dimension.

### 4.5 Le « plafond » était du sous-entraînement (correction)

Avant d'attribuer le plafond ~0,79 à la cellule, on a vérifié le **budget d'entraînement** : R0 et
variantes ré-entraînés à **10 000 itérations + cosine + 50 000 exemples** (vs 3000 / 15k), budget égal.

| modèle | in-dist @3000 | in-dist @10k+ | ood9 @10k+ |
|---|---|---|---|
| R0 | 0,780 | **0,951** | 0,884 |
| MultiStat | 0,791 | **0,959** | 0,911 |
| MultiStat + résidual | — | **0,962** | 0,909 |
| Linear (composition linéaire) | — | 0,378 | 0,335 |
| GRC (fold binaire) | 0,450 | 0,456 | 0,391 |

Le MLP passe de 0,78 à **0,95** : le plafond venait de l'entraînement, **pas de la tâche** — ce qui
corrige l'interprétation de §4.4. Les enseignements relatifs tiennent toutefois : MultiStat ≥ R0
(net en OOD) ; la composition **linéaire échoue** (MED/`mod` non-linéaires) ; le **résidual est neutre** ;
et la cellule GRC **reste à ~0,45 même à 10k** — son déficit est intrinsèque à la cellule (fold
séquentiel ≠ opérateurs symétriques), pas un manque d'entraînement.

### 4.6 Une contribution positive : l'agrégateur appris (Set-Attention)

Puisque enrichir l'agrégation *soft* aidait (MultiStat), on a poussé l'idée : au lieu de statistiques
fixées à la main, une cellule où **H requêtes apprises attendent sur les enfants** (façon PMA /
Set Transformer) et extraient H résumés soft, invariants par permutation. À budget égal, bien entraînée :

| modèle | in-dist | ood9 (profondeur 9) |
|---|---|---|
| R0 (moyenne) | 0,951 | 0,884 |
| MultiStat (stats fixes) | 0,959 | 0,911 |
| **SAA (agrégat appris, 4 têtes)** | **0,988** | **0,971** |

L'agrégation **apprise bat nettement les statistiques fixes** (+0,03 in-dist, **+0,06 en OOD**), de
façon robuste (3 seeds, ±0,002). Le gain est surtout **compositionnel** (extrapolation en profondeur),
ce qui suggère une vraie règle de composition apprise. Sous-résultat notable : la variante *hybride*
(têtes apprises + max/min explicites) est **moins bonne** (0,970) que le pur appris — encore une fois,
*un composant rigide superflu nuit*.

### 4.7 Apprendre la structure elle-même : trois échecs, un diagnostic unifié

Restait la frontière la plus dure : recevoir une **séquence plate** (sans arbre) et apprendre la
composition. Trois tentatives, toutes par fusion binaire « easy-first » différentiable :
(1) straight-through glouton : 0,469 ; (2) + Gumbel et recuit de température : 0,436 ;
(3) + **supervision de la structure gold** (teacher forcing) : 0,411 en parsing libre,
**0,454 en plafond teacher-forcé** (structure parfaite fournie à l'évaluation). Bornes : fold fixe
gauche-droite 0,420 ; cellule GRC sur arbre gold 0,456 ; SAA sur arbre gold **0,988**.

Le motif est sans ambiguïté : la supervision fait bien apprendre la structure (parsing libre ≈ plafond
teacher-forcé), mais **le plafond lui-même (~0,45) est identique pour toute composition binaire** —
fold fixe, GRC, glouton, Gumbel, supervisé. Le goulot n'a jamais été la découverte de structure :
c'est la **binarisation**. Nos opérateurs sont K-aires et symétriques, et MED n'est pas décomposable
par paires (une médiane de médianes est fausse) : un fold binaire devrait transporter le multiset
entier dans son état, ce qui ne s'apprend pas dans ce régime. C'est précisément pourquoi l'agrégation
**K-aire** (SAA) atteint 0,988 là où toute composition binaire plafonne à ~0,45 — les échecs du GRC
et du parseur latent sont un seul et même échec.

Ce diagnostic **désigne une architecture que nous n'avions pas essayée** : un parseur à structure latente qui fusionne
des **groupes K-aires** (opérateur + span d'opérandes) via l'agrégateur appris, au lieu de paires.
Hypothèse dérivée de nos données (« œuf et poule ») : l'induction de structure non supervisée échouait
peut-être *parce que* le composeur binaire ne savait rien calculer — aucun signal ne récompensait une
bonne fusion ; avec un composeur K-aire qui calcule, le signal de structure deviendrait informatif.
Protocole pré-enregistré avant expérience : `SAA_PARSER_PLAN.md`.

### 4.8 Le SAA-Parser : la synthèse des deux frontières, testée et validée

**Étape 1 (causalité du diagnostic).** Sur le nouveau parseur (fenêtre K+2, composeur SAA), structure
donnée (`sup`) : **0,925 ± 0,009** — contre ~0,45 pour toute composition binaire. Le diagnostic
« binarisation = coupable » est confirmé causalement, bien au-delà du seuil ≥0,9 fixé d'avance.
En parsing libre from-scratch : 0,660 ± 0,021 — largement au-dessus du mur binaire, mais loin du plafond.

**Ablation (fenêtre K-aire vs composeur).** Même parseur, cellule interchangeable, mode libre :
vanilla (mean-pool) 0,634 ± 0,018, multistat 0,628 ± 0,003, SAA 0,660 ± 0,021 — **quasi indiscernables**.
Contrairement au régime supervisé (SAA ≫ binaire), la qualité du composeur ne se manifeste pas quand
la structure doit être découverte : les erreurs de parsing dominent. Le saut binaire→K-aire vient
d'abord de la **fenêtre elle-même** (regrouper K opérandes d'un coup), pas spécifiquement de
l'attention apprise.

**Curriculum (le résultat-phare).** Geler la structure en teacher forcing pendant les 30% premiers pas
d'entraînement, puis sevrage linéaire vers le parsing libre : **0,909 ± 0,017**, soit 98,3% du plafond
supervisé (0,925), contre 0,660 pour le free-from-scratch et ~0,45 pour toute composition binaire
(3 seeds, σ=0,017). Ce curriculum a été conçu après avoir vu le 0,660 du mode libre ; il n'était pas dans le plan
pré-enregistré (voir §6). Sous cette réserve, il soutient nettement l'hypothèse œuf-poule : laisser le composeur devenir
compétent *avant* de lui confier la découverte de structure débloque presque tout le potentiel — une
architecture latente K-aire, entraînée entièrement par gradient (sans recherche discrète), résout ici
la tâche. (Les travaux sur les arbres latents recourent souvent à une recherche discrète comme le beam
search ; nous n'avons lancé aucune baseline de ce type, voir §6.) Bonus : le curriculum améliore aussi l'OOD
(ood9 0,459 vs 0,278 pour le free) — meilleure généralisation, pas seulement meilleur score in-dist.
Généralité : sur ListOps complet (4 opérateurs, pas seulement SM/MED), le parseur libre atteint
0,753 ± 0,007 — le mécanisme n'est pas spécifique au sous-ensemble d'opérateurs durs utilisé pour
le diagnostic.

**Limite qui persistait — résolue en §4.9.** À ce stade, les meilleurs modes perdaient ~45–50%
d'accuracy entre in-dist et profondeur 9 (OOD).

### 4.9 Généralisation : positions, opérateurs, arité

Trois tests de généralisation puis une configuration combinée (plans pré-enregistrés :
`VARIABLE_ARITY_PLAN.md`, `NIGHT3_PLAN.md`, `NIGHT4_PLAN.md` ; figure `generalization_summary.svg`).
Tous les runs de cette section sont en mode `sup` ou `curriculum`, 3 seeds :

**(a) Sans embeddings positionnels — le problème ouvert résolu.** Hypothèse : le scan de fenêtres est
*local* ; les positions absolues, sous-entraînées aux longueurs OOD, empoisonnent l'extrapolation.
Config identique à §4.8, seul `--no_pos` change : sup passe de 0,925/0,435 (in/ood9) à
**0,968 ± 0,002 / 0,930 ± 0,013** ; curriculum de 0,909/0,459 à **0,965 / 0,917**. Retirer les
positions améliore *même l'in-dist* et fait quasiment disparaître la dégradation en profondeur
(−0,04 seulement entre in-dist et profondeur 9). Encore un composant superflu qui nuisait.

**(b) Opérateurs jamais vus (MODE = plus fréquent, RNG = max−min).** Sémantiques d'une autre nature
(comptage, combinaison de statistiques d'ordre) : sup/curriculum atteignent **0,993–0,995** in-dist,
0,81–0,85 à profondeur 9. Le mécanisme n'est pas spécifique aux opérateurs du diagnostic.

**(c) Arité variable [2,7] — d'abord une limite, puis un diagnostic prouvé.** Le plafond chute à
~0,61 *même avec structure gold* (in_tf = in), et un contrôle no_pos montre que ce n'est **pas**
positionnel (0,615 ≈ 0,611). Le diagnostic désigné par l'architecture : l'attention masquée calcule
des *moyennes pondérées*, or SM = somme exige le *nombre* d'opérandes — à K fixe, somme = K×moyenne
(constante apprenable) ; à K variable, le compte est détruit par le softmax. **Réinjecter un embedding
du compte d'enfants valides** (+13k params) fait passer le plafond de 0,61 à **0,90 ± 0,01**
(0,80 en profondeur 9). Troisième fix désigné par diagnostic du projet (hypernet → poids directs ;
OOD → no_pos ; arité → count).

**(d) Le chiffre-vitrine.** Tout combiné — un seul modèle de 192k paramètres, six opérateurs mélangés
(MAX, MIN, MED, SM, MODE, RNG), sans positions, entraîné par curriculum sur séquences plates :
**0,933 ± 0,004 in-dist, 0,915 ± 0,002 à profondeur 9** (généralisation quasi plate, −0,018). Le
curriculum y égale le plafond supervisé. Sa robustesse est vérifiée : curr_hold ∈ {0,1 ; 0,3 ; 0,5}
donne 0,965–0,969 à l'identique — c'est le *sevrage progressif* qui compte, pas son réglage.

## 5. Discussion

À travers quatre architectures structurées indépendantes (cosmologie ; hypernet ; histogramme « optimal » ;
et même la baseline avant enrichissement), un motif unique ressort : **les priors rigides et discrets
perdent contre des features soft et haute-dimension fournies à un MLP générique.** Le seul gain au-dessus
de la baseline vient d'un enrichissement soft, pas d'une géométrie imposée. La leçon mécaniste la plus
réutilisable concerne les **hypernetworks à génération** : ils peuvent coûter en *capacité effective*
(les poids vivent sur une variété basse-dimension), d'autant plus que le nombre de jeux de poids
demandés croît — un coût caché derrière la promesse du « partage massif de paramètres ».

Sur le plan épistémique, le projet illustre la prudence requise face aux architectures « élégantes » :
deux prédictions de victoire fondées sur l'élégance (la cosmologie, puis notre histogramme) se sont
révélées fausses ; les améliorations réelles sont venues d'hypothèses *ancrées dans les données* et
testées *sans pari* — jusqu'au SAA-Parser, où l'architecture elle-même a été **désignée** par un
diagnostic mécaniste plutôt qu'imaginée. L'ablation §4.8 ajoute une nuance à la thèse « soft bat
rigide » : en régime de structure *inconnue*, la qualité du composeur ne suffit pas à se manifester —
elle a besoin d'un curriculum qui la protège de l'exposure bias le temps qu'elle devienne compétente.
Le couple « bon composeur + bon curriculum » compte plus que le composeur seul.

La section 4.9 répète le même schéma trois fois : un échec est localisé, le diagnostic désigne un
correctif précis, et le correctif est testé contre un seuil fixé d'avance. Deux de ces correctifs sont
de signe opposé. Retirer les embeddings positionnels est une *soustraction* : le scan de fenêtres est
local, et une information de position absolue, mal apprise aux longueurs jamais vues, ne faisait que
nuire. Réinjecter le compte d'opérandes est une *addition* : l'attention normalisée par softmax est
une moyenne, et une moyenne ne suffit pas à calculer une somme dès que le nombre de termes varie. La
règle « un composant superflu nuit » ne dispense donc pas de vérifier ce que le composeur peut
réellement représenter.

## 6. Limites

- **Aucune comparaison à une méthode d'arbre latent publiée.** Nos références binaires (§4.7) sont nos
  propres implémentations ; nous n'avons pas entraîné, au même budget, un parseur latent de la
  littérature (par exemple un Tree-LSTM à Gumbel-softmax). Sans cette comparaison, on peut dire que le
  parseur K-aire fonctionne, pas qu'il est meilleur que l'existant. Aucune des briques n'est nouvelle
  prise isolément (arbres latents, attention sur ensemble, passage progressif du teacher forcing au
  mode libre) ; l'apport est leur combinaison et le diagnostic qui la motive. Nous n'avons pas fait de
  revue systématique de la littérature : nous n'affirmons pas que le parsing latent K-aire ou ce
  curriculum soient nouveaux.
- **Une seule tâche, synthétique.** Des variantes de ListOps produites par notre propre générateur
  (arité fixe, petits arbres, profondeur contrôlée). Ce n'est pas le benchmark ListOps de Long Range
  Arena, et nos chiffres ne sont pas comparables aux scores ListOps publiés. Rien n'est testé sur des
  données réelles (AST de code, molécules, langage) ; le SAA-Parser rend ce terrain plus plausible, pas
  acquis.
- **Arité bornée et annoncée.** L'arité variable (§4.9c) passe par une fenêtre canonique de taille
  Kmax + 2 remplie de tokens PAD : le modèle voit donc où sont les emplacements vides. Avec le compte,
  le plafond remonte à 0,90, soit encore 0,06 sous le régime à arité fixe (0,965). Ces runs utilisent
  aussi un MLP plus étroit (148–161k paramètres contre 192k), ce qui interdit une comparaison stricte.
- **Mode libre non re-testé sans positions.** Le 0,660 du parsing libre sans curriculum (§4.8) date
  de la configuration avec positions ; l'écart libre/curriculum sans positions n'est pas mesuré.
- **Curriculum = supervision de structure.** Le curriculum utilise l'arbre gold pendant
  l'entraînement ; seul le mode libre s'en passe. « Sans recherche discrète » ne veut pas dire « sans
  supervision ».
- **Petite échelle** (modèles ~0,15–1,1M, RTX 3070). Les phénomènes pourraient différer à grande échelle.
- Deux limites antérieures sont levées en §4.9 : la dégradation en profondeur (artefact des
  embeddings positionnels ; ood9 passe de 0,46 à 0,92) et la spécificité aux opérateurs SM/MED.
- **Le curriculum n'était pas pré-enregistré.** Le plan écrit avant l'expérience (`SAA_PARSER_PLAN.md`)
  testait l'hypothèse œuf-poule par le mode libre, avec un seuil de 0,85 ; le résultat (0,660) tombe dans
  la case « structure partiellement apprise ». Le curriculum a été conçu après avoir vu ce chiffre. Il
  soutient l'hypothèse, mais c'est une expérience a posteriori, à lire comme telle. Les tests de §4.9
  (positions, opérateurs, compte, robustesse) ont, eux, des seuils écrits avant les runs.
- **Nombre de seeds et variance.** 3 seeds partout sauf : Linear et GRC à 10k, le fold fixe l2r, et les
  ré-entraînements Etz4/histogramme à 10k (2 seeds chacun). Les ± sont des écarts-types de population
  sur ces seeds, donc de simples indicateurs de dispersion. Deux résultats sont bruités : Etz4 sans
  hypernet (0,752 ± 0,069) et le parseur binaire glouton (0,469 ± 0,054).
- **Tournoi texte (§4.1) : un seul run** de 1000 itérations par modèle, sans seeds multiples, et à
  budgets inégaux (Transformer 816k, Etz 1,1M). C'est un constat de départ, pas une mesure contrôlée.
- **Le gain de MultiStat est modeste** (+0,008 in-dist à 10k, +0,027 à profondeur 9).
- Etz4 et l'histogramme ont été ré-entraînés à 10k itérations (0,469 et 0,517, 2 seeds) : leur déficit
  ne vient pas du budget d'entraînement.

## 7. Conclusion

La Kabbale inspire de bons *principes* (la hiérarchie aide quand la donnée est structurée), mais ne
produit pas, en l'état, une *architecture* compétitive : le hypernetwork générateur la pénalise, et une
fois corrigée, sa structure cosmologique est neutre face à un MLP. En poursuivant la question « quelle
architecture cette tâche réclame-t-elle vraiment », deux contributions positives émergent, chacune
désignée par le diagnostic de la précédente plutôt que par l'élégance : (1) sur arbre donné,
l'**agrégateur appris par attention sur ensemble (SAA)** atteint 0,988 in-dist / 0,971 OOD à budget
égal ; (2) sur **séquence plate**, la découverte que toute composition *binaire* plafonne à ~0,45
(la médiane n'est pas décomposable par paires) désigne un **parseur à structure latente K-aire**, qui,
entraîné par un **curriculum** protégeant le composeur de l'exposure bias, atteint **0,909 — 98,3% du
plafond obtenu avec la structure donnée**, entièrement par gradient, sans recherche discrète. Les
tests de généralisation corrigent ensuite ses deux faiblesses mesurées : sans embeddings positionnels,
la dégradation en profondeur disparaît (0,965 in-distribution, 0,917 à profondeur 9) ; avec le compte
d'opérandes, l'arité variable remonte de 0,61 à 0,90. La configuration combinée — six opérateurs, un
modèle de 192k paramètres — atteint **0,933 / 0,915**. C'est un résultat négatif honnête sur la
Kabbale, mécaniquement diagnostiqué, qui débouche sur deux architectures positives et une règle de
méthode : *chercher ce que le diagnostic désigne, pas ce qui est élégant — et quand un composant
capable échoue en régime non supervisé, soupçonner l'ordre d'apprentissage avant l'architecture
elle-même.* Ce que ce travail ne montre pas encore : que le parseur K-aire fait mieux qu'un parseur
latent de la littérature, et qu'il tient hors d'une tâche synthétique.

---

*Code, données brutes (JSONL) et figures : voir le dépôt (`listops_experiment/`, `RESULTS.md` et son
index des fichiers de données). Plans pré-enregistrés : `SAA_PARSER_PLAN.md`, `VARIABLE_ARITY_PLAN.md`,
`NIGHT3_PLAN.md`, `NIGHT4_PLAN.md`. Figures : `ood_curves.svg`, `ood_curves_hard.svg`,
`ablation_autopsy.svg`, `training_budget.svg`, `learned_aggregator.svg`, `saa_parser_curriculum.svg`,
`generalization_summary.svg`. Version anglaise : `PAPER_EN.md`.*
