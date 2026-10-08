# Plan 2 — A LEARNED aggregator (set attention / PMA)

*English translation of the plan written before the runs (original: `docs/fr/plans/OVERNIGHT_PLAN_B.md`,
committed 2026-07-01 02:46; the results were committed 2026-07-01 09:08). Text in [brackets] was added
in translation. Results: `RESULTS.md`, phase 9.*

Detached, checkpointed per (seed, model). Output: `results_saa.jsonl`.

## The idea

MultiStat won with 3 statistics *fixed by hand* (`mean;max;min`). What if the model **learned by
itself** which statistics to extract? The `SetAggCell` cell uses H **learned queries** that attend
(softmax) over the children → H soft, permutation-invariant summaries (in the manner of PMA / Set
Transformer). It generalises MultiStat (mean = uniform attention, max ≈ peaked attention), while
staying **soft + high-dimensional** — the axis our data supports.

## Configuration (identical to the phase 8 sweep → directly comparable)

Hard variant (SM, MED); gold tree; K=5; cap 120; training depths 1-4, OOD 5/7/9; budget ~207k;
**10k iterations + cosine + 50k examples**; 3 seeds; data_seed 1234.

## Variants tested

| kind | what is tested |
|---|---|
| SAA_h2 | 2 learned heads |
| SAA_h4 | 4 learned heads |
| SAA_h8 | 8 learned heads (do more learned statistics help?) |
| SAA_hybrid | 4 learned heads **+** explicit [max;min] (learned + hand-picked) |

Compared with (already measured at the SAME budget, `results_overnight.jsonl`):
- **R0** (mean-pool): in 0.951 · ood9 0.884
- **MultiStat** (fixed [mean;max;min]): in 0.959 · ood9 0.911

## Reading grid (pre-registered)

- **SAA > MultiStat** → *learned* aggregation beats *fixed* statistics: a real contribution.
- **SAA ≈ MultiStat** → the 3 fixed statistics already captured the essentials; attention adds
  nothing (an honest result, consistent with "simple is enough").
- **SAA < MultiStat** → pure attention struggles to reproduce max/min; then **SAA_hybrid** should
  close the gap (learned heads + guaranteed max/min) → a clear diagnosis.
- Effect of the number of heads (h2/h4/h8) → how many distinct statistics the task calls for.

## Analysis steps

1. aggregate (mean ± σ), compare with R0/MultiStat, figure;
2. results log (phase 9) and paper (section "learned aggregation");
3. if SAA wins → move on to learning the structure itself, reusing SAA as the cell.

[Outcome: the first row of the grid. SAA_h4 = 0.988 in / 0.971 ood9; the hybrid is worse (0.970).]
