# Plan du run nocturne autonome (carte blanche ~8 h)

Lancé en détaché (PID variable), checkpointé par (seed, modèle), survit aux fermetures de session.
Sortie : `scratchpad/full_results_overnight.jsonl` (copiée vers `results_overnight.jsonl` au dépouillement).

## But

Répondre aux questions de l'advisor (GLM-5.2), en priorité **#1 : le budget d'entraînement est-il
le goulot ?**, plutôt que de tester une énième cellule à l'aveugle.

## Config commune

Variante DURE (SM, MED) ; arbre gold ; arité K=5 ; cap 120 nœuds ; profondeurs train 1-4, OOD 5/7/9 ;
**budget ~207k** (sauf Linear) ; **10 000 itérations** (≈3,3× nos 3000) ; **cosine schedule + warmup 300** ;
**50 000 exemples** (vs 15k, pour éviter l'overfit à plus d'itérations) ; 3 seeds.

## Modèles (axes croisés)

| kind | axe testé | question |
|---|---|---|
| R0 | baseline | plafond d'entraînement (vs R0@3000=0,780) |
| MultiStat | pooling [mean;max;min] | le gain Phase 7 tient-il à plus long budget ? |
| R0_resid | résidual (#3) | le compounding est-il la cause ? |
| MultiStat_resid | pooling + résidual | additivité des gains ? |
| Linear | linéarité (#2) | la non-linéarité de composition nuit-elle à l'OOD ? |
| GRC | cellule gated (#5) | la cellule SOTA aide-t-elle dans notre régime ? |

## Grille de lecture (pré-enregistrée)

- **R0@10k ≫ R0@3000 (0,78)** → le goulot était l'entraînement ; relire toutes les conclusions
  « cellule » à ce budget.
- **R0@10k ≈ 0,80** → la cellule/représentation/tâche plafonne ; alors les axes (résidual, linéaire,
  GRC) départagent ce qui aide.
- Comparer les axes ENTRE EUX à budget/itérations égaux → quel levier porte le gain.

## Au dépouillement (matin / session suivante)

1. `cp scratchpad/full_results_overnight.jsonl listops_experiment/results_overnight.jsonl`
2. agréger (moyenne ± écart-type) par kind ; comparer à R0@3000 (0,780) et au sweep dur stocké.
3. mettre à jour RESULTS.md (Phase 8) + PAPER.md (training-budget + ablations d'axes) + commit.
