# Plan 1 — Is the training budget the bottleneck?

*English translation of the plan written before the runs (original: `docs/fr/plans/OVERNIGHT_PLAN.md`,
committed 2026-06-30 03:45; the results were committed 2026-06-30 13:49). Text in [brackets] was added
in translation. Results: `RESULTS.md`, phase 8.*

Run detached, checkpointed per (seed, model), so that it survives session restarts.
Output: `results_overnight.jsonl`.

## Goal

Answer the questions raised by an external reviewer [an AI assistant used as a second opinion], with
priority **no. 1: is the training budget the bottleneck?**, rather than testing yet another cell
blindly.

## Common configuration

HARD variant (SM, MED); gold tree; arity K=5; cap of 120 nodes; training depths 1-4, OOD 5/7/9;
**budget ~207k** (except Linear); **10,000 iterations** (about 3.3× our 3,000); **cosine schedule +
warmup 300**; **50,000 examples** (vs 15k, to avoid overfitting at more iterations); 3 seeds.

## Models (crossed axes)

| kind | axis tested | question |
|---|---|---|
| R0 | baseline | training ceiling (vs R0@3000 = 0.780) |
| MultiStat | [mean;max;min] pooling | does the phase 7 gain hold at a longer budget? |
| R0_resid | residual | is error compounding the cause? |
| MultiStat_resid | pooling + residual | do the gains add up? |
| Linear | linearity | does a non-linear composition hurt OOD? |
| GRC | gated cell | does this kind of cell help in our regime? |

## Reading grid (pre-registered)

- **R0@10k ≫ R0@3000 (0.78)** → the bottleneck was training; re-read every "cell" conclusion at this
  budget.
- **R0@10k ≈ 0.80** → the cell / representation / task is the ceiling; then the axes (residual, linear,
  GRC) tell what helps.
- Compare the axes WITH EACH OTHER at equal budget and iterations → which lever carries the gain.

## Analysis steps

1. aggregate (mean ± standard deviation) per kind; compare with R0@3000 (0.780) and with the stored
   hard-variant sweep;
2. update the results log (phase 8) and the paper (training budget + axis ablations).

[Outcome: the first row of the grid. R0 went from 0.780 to 0.951. The run stopped at 16 of 18 planned
runs, so Linear and GRC have 2 seeds.]
