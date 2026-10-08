# Nuit #2 — Frontière B : inventer un agrégateur APPRIS (Set-Attention / PMA)

Détaché, checkpointé par (seed, modèle), survit aux fermetures de session.
Sortie : `scratchpad/full_results_saa.jsonl` → `results_saa.jsonl` au dépouillement.

## L'idée inventée

MultiStat a gagné avec 3 statistiques *fixées à la main* (`mean;max;min`). Frontière B : et si le
modèle **apprenait lui-même** quelles statistiques extraire ? La cellule `SetAggCell` utilise H
**requêtes apprises** qui attendent (softmax) sur les enfants → H résumés soft, invariants par
permutation (façon PMA / Set Transformer). Généralise MultiStat (mean = attention uniforme,
max ≈ attention piquée), tout en restant **soft + haute-dim** — l'axe que nos données valident.

## Config (identique au sweep Phase 8 → directement comparable)

Variante dure (SM,MED) ; arbre gold ; K=5 ; cap 120 ; train prof. 1-4, OOD 5/7/9 ; budget ~207k ;
**10k iters + cosine + 50k données** ; 3 seeds ; data_seed 1234.

## Variantes testées

| kind | ce qu'on teste |
|---|---|
| SAA_h2 | 2 têtes apprises |
| SAA_h4 | 4 têtes apprises |
| SAA_h8 | 8 têtes apprises (plus de statistiques apprises aide-t-il ?) |
| SAA_hybrid | 4 têtes apprises **+** [max;min] explicites (learned + hand-picked) |

Comparés à (déjà mesurés au MÊME budget, `results_overnight.jsonl`) :
- **R0** (mean-pool) : in 0,951 · ood9 0,884
- **MultiStat** ([mean;max;min] fixes) : in 0,959 · ood9 0,911

## Grille de lecture (pré-enregistrée)

- **SAA > MultiStat** → l'agrégation *apprise* bat les statistiques *fixes* : vraie contribution
  (« learned aggregation > fixed order-statistics for recursive computation »).
- **SAA ≈ MultiStat** → les 3 stats fixes captaient déjà l'essentiel ; l'attention n'ajoute rien
  (résultat honnête, cohérent avec « le simple suffit »).
- **SAA < MultiStat** → l'attention pure a du mal à reproduire max/min ; alors **SAA_hybrid** devrait
  combler (têtes apprises + max/min garantis) → diagnostic net.
- Effet du nombre de têtes (h2/h4/h8) → combien de statistiques distinctes la tâche réclame.

## Au dépouillement

1. `cp scratchpad/full_results_saa.jsonl listops_experiment/results_saa.jsonl`
2. agréger (moyenne ± σ), comparer à R0/MultiStat, graphe.
3. RESULTS.md (Phase 9) + PAPER.md (section « agrégation apprise ») + commit.
4. Si SAA gagne → enchaîner Frontière A (structure soft) en réutilisant SAA comme cellule.
