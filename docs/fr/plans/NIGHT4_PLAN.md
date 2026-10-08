# Nuit #4 — 10h carte blanche (plan écrit avant expérience)

Enchaînement automatique (night4_chain.bat, attend le checkpoint du bloc précédent).
Préalable en cours : `results_vararity_nopos.jsonl` (l'arité variable chute-t-elle à cause des
positions ?) — signal précoce : non (~0,60 ≈ 0,61 avec positions), la limite semble réelle.

## Bloc A — Diagnostic count-feature (la dernière limite, attaquée mécaniquement)

**Hypothèse désignée par l'architecture** : l'attention masquée calcule des *moyennes pondérées* ;
or SM = somme exige le *nombre* d'opérandes. À K fixe, somme = K×moyenne (constante apprenable) ;
à K variable, le compte est détruit par le softmax → le composeur ne peut pas calculer SM.
**Fix** : réinjecter un embedding du compte d'enfants valides dans le composeur (`--count`).
Config : vararity, no_pos + count, sup+curr ×3 seeds → `full_results_var_count.jsonl`.

| observation | conclusion |
|---|---|
| sup ≥ ~0,85 | diagnostic **prouvé** : la limite d'arité variable était le compte perdu → levée |
| sup ~0,7-0,85 | le compte est une partie de l'explication, pas toute |
| sup ≈ 0,61 inchangé | diagnostic réfuté — la difficulté est ailleurs (à documenter) |

## Bloc B — Config-vitrine : UN modèle, SIX opérateurs, séquence plate

Tout ce qui a été appris, combiné : parseur K=5 fixe + no_pos + curriculum, sur le mélange
MAX,MIN,MED,SM,MODE,RNG (6 opérateurs, dont 2 jamais utilisés ensemble avec les 4 autres).
sup+curr ×3 seeds → `full_results_capstone.jsonl`. C'est le chiffre-vitrine du papier :
« un seul modèle apprend structure + calcul de 6 opérateurs depuis des séquences plates ».
Attendu (si les acquis composent) : ≥0,9 in-dist, OOD fort grâce à no_pos. Aucun pari — on mesure.

## Bloc C — Robustesse du curriculum (le 30% est-il magique ?)

curr_hold = 0,1 et 0,5 (vs 0,3 utilisé partout), parseur K=5 fixe, SM+MED, no_pos, curr ×3 seeds
→ `full_results_ch01.jsonl`, `full_results_ch05.jsonl`.
Si les trois valeurs donnent ~pareil → le curriculum est robuste (pas un hyperparamètre fragile).
Si 0,1 s'effondre → il faut vraiment laisser le composeur apprendre d'abord (renforce l'œuf-poule).

## Pendant les runs (CPU)

Traduction anglaise de PAPER.md (`PAPER_EN.md`) — préparation à une soumission éventuelle.
Dépouillement au fil de l'eau, figures si utile, commits par bloc, rapport final au matin.
