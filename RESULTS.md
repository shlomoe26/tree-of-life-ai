# Results

Every table below is computed from the raw files in `listops_experiment/results_*.jsonl` (one JSON
line per training run). The original experiment log, written in French as the work went along, is in
`docs/fr/RESULTS_FR.md`; this file is a restructured English version with the numbers recomputed from
the data.

## Reading the tables

- **Task.** ListOps: nested expressions over digits 0-9; the answer is one digit, so chance is 0.10.
  Unless stated otherwise: fixed arity K=5, at most 120 nodes per tree, training depths 1-4.
- **in** = accuracy on held-out expressions of depths 1-4. **ood5 / ood7 / ood9** = accuracy on
  expressions of depth 5, 7, 9, deeper than anything seen in training.
- **in_tf** (parser tables only) = accuracy when the correct structure is supplied at test time
  (teacher forcing). All other parser columns are *free* parsing: the model picks the structure itself.
- **± ** is the population standard deviation over seeds (n = 3 unless a table says otherwise). With
  three seeds it is an indication of spread, not a confidence interval. No significance tests were run.
- **"Hard variant"** = operators SM (sum mod 10) and MED (median) only.
- **Two training regimes.** Phases 4-7 use 3,000 iterations and 15,000 training examples. From
  phase 8 on: 10,000 iterations, cosine schedule with warmup, 50,000 examples. The exact options of
  each run are in the plans; the scripts' defaults are not always the values used.

## Phases 1-3 — Text tournament (not a controlled comparison)

Character-level language modelling on Tiny Shakespeare, 1,000 iterations, **one run per model**,
validation loss (lower is better). The models do not have the same number of parameters.

| model | parameters | validation loss |
|---|---|---|
| Bigram | — | 3.45 |
| Etz, 1 world | 1.10M | ≈ 2.50 |
| Etz, 4 worlds + Tzimtzum | 1.13M | 2.49 |
| Etz, 4 worlds + one attention block | 1.18M | 2.19 |
| Transformer (4 layers) | 0.82M | 1.80 |

These numbers come from single runs of `run_sweep.py` during the early part of the project and were
not stored in a results file; the bigram figure is from a separate run (the script as shipped does
not include it). They motivated the rest of the work and should not be read as more than that. The
useful observation is qualitative: 1 world ≈ 4 worlds, because `models/etz_unified.py` copies the same
vector onto all 10 nodes (`repeat(1, 10, 1)`), so the graph never receives any structure.

## Phase 4 — The Etz as a recursive cell, tree given

All models ≈ 1.1M parameters, 3,000 iterations, 3 seeds. Figure: `fig1_etz_vs_mlp.svg`.

All four original operators (`results_easy.jsonl`):

| model | params | in | ood5 | ood6 | ood7 |
|---|---|---|---|---|---|
| R0 (mean-pool MLP) | 1,126,554 | 0.745 ± 0.009 | 0.698 ± 0.014 | 0.711 ± 0.007 | 0.757 ± 0.007 |
| Etz, 1 world | 1,075,790 | 0.536 ± 0.001 | 0.504 ± 0.003 | 0.520 ± 0.004 | 0.523 ± 0.009 |
| Etz, 4 worlds + Tzimtzum | 1,110,938 | 0.664 ± 0.008 | 0.638 ± 0.004 | 0.652 ± 0.011 | 0.674 ± 0.013 |

Hard variant (`results_hard.jsonl`):

| model | in | ood5 | ood7 | ood9 |
|---|---|---|---|---|
| R0 | 0.737 ± 0.007 | 0.538 ± 0.015 | 0.557 ± 0.019 | 0.617 ± 0.016 |
| Etz, 1 world | 0.363 ± 0.007 | 0.305 ± 0.001 | 0.314 ± 0.002 | 0.326 ± 0.005 |
| Etz, 4 worlds + Tzimtzum | 0.466 ± 0.024 | 0.387 ± 0.017 | 0.386 ± 0.016 | 0.397 ± 0.025 |

Once real structure is fed in, 4 worlds beat 1 world (+0.13 and +0.10), but the Etz stays below the
flat MLP, and the gap grows on the hard variant (−0.08 → −0.27).

## Phase 5 — Autopsy: which component costs the accuracy?

Hard variant, 3,000 iterations, 3 seeds (`results_ablation.jsonl`, `results_followup.jsonl`).
Figure: `fig2_autopsy.svg`.

| model | params | in | ood9 |
|---|---|---|---|
| Etz 4 worlds, hypernetwork (phase 4) | 1,110,938 | 0.466 ± 0.024 | 0.397 ± 0.025 |
| Etz 4 worlds, learned gate on the Tzimtzum | 1,110,941 | 0.504 ± 0.055 | 0.425 ± 0.030 |
| Etz 4 worlds, direct weights (no hypernetwork) | 205,018 | 0.752 ± 0.069 | 0.629 ± 0.069 |
| Etz 1 world, direct weights | 47,086 | 0.359 ± 0.005 | 0.329 ± 0.005 |
| R0 at matched budget | 208,554 | 0.780 ± 0.008 | 0.644 ± 0.018 |

Replacing the weight-generating hypernetwork with ordinary parameters gives +0.29 with about 5× fewer
parameters. At 1 world the two are equal (0.363 vs 0.359); the harm appears with depth. Note the large
spread of the direct-weights run (± 0.069). At matched budget the flat MLP is at least as good.

## Phase 6 — A cell we predicted would win, and did not

A histogram cell: project each child to a distribution over the 10 values, sum, read out with an
operator-conditioned head. Prediction written before the run: beat R0 and remove the OOD drop.
Hard variant, 3,000 iterations, 3 seeds (`results_histogram.jsonl`).

| model | params | in | ood9 |
|---|---|---|---|
| Histogram | 103,636 | 0.463 ± 0.018 | 0.401 ± 0.005 |
| Histogram, budget matched | 218,260 | 0.458 ± 0.014 | 0.392 ± 0.003 |
| R0 (phase 5) | 208,554 | 0.780 ± 0.008 | 0.644 ± 0.018 |

The prediction was wrong.

## Phase 7 — Multi-statistic pooling

Replace `mean(children)` with `[mean ; max ; min]`. Hard variant, 3,000 iterations, 3 seeds
(`results_multistat.jsonl`).

| model | params | in | ood5 | ood7 | ood9 |
|---|---|---|---|---|---|
| MultiStat | 207,434 | 0.791 ± 0.003 | 0.605 ± 0.010 | 0.614 ± 0.002 | 0.670 ± 0.006 |
| R0 (phase 5) | 208,554 | 0.780 ± 0.008 | 0.577 ± 0.018 | 0.601 ± 0.007 | 0.644 ± 0.018 |

## Phase 8 — The "0.79 ceiling" was under-training

Same cells, 10,000 iterations + cosine + 50,000 examples (`results_overnight.jsonl`,
`results_grc3k.jsonl`). Plan: `plans/OVERNIGHT_PLAN.md`. Figure: `fig3_training_budget.svg`.

| model | seeds | params | in | ood5 | ood7 | ood9 |
|---|---|---|---|---|---|---|
| R0 | 3 | 208,554 | 0.951 ± 0.004 | 0.853 ± 0.021 | 0.851 ± 0.014 | 0.884 ± 0.005 |
| MultiStat | 3 | 207,434 | 0.959 ± 0.003 | 0.892 ± 0.006 | 0.873 ± 0.010 | 0.911 ± 0.003 |
| R0 + residual | 3 | 208,555 | 0.949 ± 0.005 | 0.852 ± 0.014 | 0.845 ± 0.009 | 0.885 ± 0.002 |
| MultiStat + residual | 3 | 207,435 | 0.962 ± 0.002 | 0.888 ± 0.007 | 0.868 ± 0.007 | 0.909 ± 0.013 |
| Linear composition | 2 | 9,802 | 0.378 ± 0.001 | 0.310 ± 0.003 | 0.317 ± 0.003 | 0.335 ± 0.002 |
| GRC-style binary fold | 2 | 208,290 | 0.456 ± 0.005 | 0.387 ± 0.009 | 0.369 ± 0.008 | 0.391 ± 0.004 |
| GRC-style binary fold, 3,000 iterations | 3 | 208,290 | 0.450 ± 0.004 | 0.370 ± 0.003 | 0.358 ± 0.005 | 0.404 ± 0.011 |

R0 goes from 0.780 to 0.951 with more training: the conclusion of phases 6-7 that "the task plateaus
near 0.79" was an artefact. The relative ordering holds. The run was stopped at 16 of 18 planned
runs, which is why Linear and GRC have 2 seeds. The GRC-style cell is our own approximation of a
gated recursive cell, not a reproduction of a published model.

## Phase 9 — Learned aggregation (SAA)

H learned queries attend over the children. Same regime as phase 8, 3 seeds (`results_saa.jsonl`).
Plan: `plans/OVERNIGHT_PLAN_B.md`. Figure: `fig4_cells.svg`.

| model | params | in | ood5 | ood7 | ood9 |
|---|---|---|---|---|---|
| SAA, 2 heads | 209,898 | 0.986 ± 0.001 | 0.970 ± 0.005 | 0.953 ± 0.007 | 0.966 ± 0.003 |
| **SAA, 4 heads** | 212,570 | **0.988 ± 0.002** | 0.968 ± 0.002 | 0.965 ± 0.006 | **0.971 ± 0.001** |
| SAA, 8 heads | 214,234 | 0.985 ± 0.003 | 0.969 ± 0.003 | 0.967 ± 0.001 | 0.969 ± 0.002 |
| SAA, 4 heads + explicit max/min | 208,074 | 0.970 ± 0.003 | 0.937 ± 0.017 | 0.917 ± 0.013 | 0.939 ± 0.008 |
| MultiStat (phase 8) | 207,434 | 0.959 ± 0.003 | 0.892 ± 0.006 | 0.873 ± 0.010 | 0.911 ± 0.003 |
| R0 (phase 8) | 208,554 | 0.951 ± 0.004 | 0.853 ± 0.021 | 0.851 ± 0.014 | 0.884 ± 0.005 |

## Phase 10 — Learning the structure with binary merges

The model now receives a flat token sequence. Hard variant, 10,000 iterations, 215,562 parameters
(`results_parser.jsonl`, `results_gumbel.jsonl`, `results_sup_w02.jsonl`, `results_sup_w10.jsonl`).
Figure: `fig5_flat_sequence.svg`.

| binary parser | seeds | in (free) | in_tf | ood5 | ood9 |
|---|---|---|---|---|---|
| fixed left-to-right fold | 2 | 0.420 ± 0.003 | — | 0.318 ± 0.002 | 0.253 ± 0.010 |
| greedy straight-through | 3 | 0.469 ± 0.054 | — | 0.335 ± 0.012 | 0.321 ± 0.005 |
| Gumbel + temperature annealing | 3 | 0.436 ± 0.006 | — | 0.336 ± 0.007 | 0.311 ± 0.005 |
| structure-supervised, loss weight 0.2 | 3 | 0.258 ± 0.026 | 0.365 ± 0.035 | 0.191 ± 0.003 | 0.180 ± 0.023 |
| structure-supervised, loss weight 1.0 | 3 | 0.411 ± 0.003 | 0.454 ± 0.000 | 0.212 ± 0.003 | 0.157 ± 0.007 |

With full structure supervision, free parsing (0.411) nearly reaches its teacher-forced ceiling
(0.454), and that ceiling equals the GRC-style fold on the gold tree (0.456). Every binary composition
we tried sits between 0.41 and 0.47, however the structure is chosen. So in these runs the limit is
our binary composer, not the discovery of structure. It is **not** a limit of binary composition in
general: published binary cells given the tree reach 98.7% to 99.95% on ListOps (`RELATED_WORK.md`,
§1). We do not know why ours failed. These are our own small binary parsers; a published latent-tree
model was not run.

Consolidation, same 10,000-iteration regime, **2 seeds** (`results_consol.jsonl`):

| model | params | in | ood9 |
|---|---|---|---|
| Etz 4 worlds, hypernetwork | 1,110,938 | 0.469 ± 0.001 | 0.415 ± 0.000 |
| Histogram, budget matched | 218,260 | 0.517 ± 0.043 | 0.457 ± 0.037 |

More training does not rescue these two cells.

## Phase 11 — SAA-Parser: K-ary groups

A parser that merges an operator and its K operands in one step, with the SAA as composer. Hard
variant, 10,000 iterations, with positional embeddings, 3 seeds. Plan: `plans/SAA_PARSER_PLAN.md`.
Figure: `fig5_flat_sequence.svg`.

| mode | file | params | in (free) | in_tf | ood5 | ood7 | ood9 |
|---|---|---|---|---|---|---|---|
| `sup` — teacher forcing throughout | `results_saaparser` | 191,562 | 0.925 ± 0.009 | 0.925 ± 0.009 | 0.670 ± 0.019 | 0.509 ± 0.035 | 0.435 ± 0.056 |
| `free` — no structure signal | `results_saaparser` | 191,562 | 0.660 ± 0.021 | 0.660 ± 0.021 | 0.329 ± 0.011 | 0.280 ± 0.011 | 0.278 ± 0.008 |
| `curr` — curriculum, 30% hold | `results_saap_curr` | 191,562 | 0.909 ± 0.017 | 0.917 ± 0.007 | 0.647 ± 0.032 | 0.504 ± 0.030 | 0.459 ± 0.041 |
| `free`, mean-pool composer | `results_saap_vanilla` | 182,730 | 0.634 ± 0.018 | — | 0.306 ± 0.025 | 0.267 ± 0.008 | 0.269 ± 0.011 |
| `free`, [mean;max;min] composer | `results_saap_ms` | 194,186 | 0.628 ± 0.003 | — | 0.314 ± 0.005 | 0.279 ± 0.006 | 0.284 ± 0.009 |
| `free`, 4 operators (MAX, MIN, MED, SM) | `results_saap_allops` | 191,562 | 0.753 ± 0.007 | — | 0.512 ± 0.018 | 0.479 ± 0.012 | 0.473 ± 0.004 |

What was pre-registered and what was not:

- Step 1 of the plan (teacher-forced accuracy ≥ ~0.9): met, 0.925. The plan called this a proof of
  the "binarisation" diagnosis; it shows that swapping our binary composer for the K-ary one removes
  the plateau, which is weaker (see `RELATED_WORK.md`, §1).
- Step 2 of the plan (free parsing ≥ ~0.85 validates the "chicken and egg" hypothesis): **not met**.
  Free parsing reaches 0.660, which the plan's grid labels "structure partially learned".
- The **curriculum** was designed after seeing the 0.660. It is a post-hoc experiment. It also uses
  the gold structure during training, so it is not an unsupervised result.
- In `free` mode the three composers are within 0.03 of each other: the gain over binary merging
  comes mainly from the K-ary window, not from learned attention.
- One `sup` run (seed 0) was first launched by mistake with a larger model (486,986 parameters). It
  was discarded and rerun at 191,562 before the numbers above were computed.

## Phase 12 — Variable arity, first attempt

Arity drawn in 2-7 per node; canonical window of Kmax+2 tokens with masked PAD tokens. Hard variant,
hidden size 192 (fixed-arity runs use 256), with positions, 3 seeds (`results_vararity.jsonl`).
Plan: `plans/VARIABLE_ARITY_PLAN.md`.

| mode | params | in | in_tf | ood5 | ood9 |
|---|---|---|---|---|---|
| `sup` | 147,466 | 0.611 ± 0.008 | 0.611 ± 0.008 | 0.390 ± 0.002 | 0.347 ± 0.011 |
| `curr` | 147,466 | 0.613 ± 0.003 | 0.613 ± 0.003 | 0.394 ± 0.008 | 0.366 ± 0.003 |

Per the pre-registered grid (sup < 0.7): this regime has a difficulty of its own. The ceiling is low
even with the correct structure, so the problem is in the computation, not the parsing.

## Phase 13 — Without positional embeddings

Identical to phase 11 except `--no_pos` (`results_nopos.jsonl`). Plan: `plans/NIGHT3_PLAN.md`.
Figure: `fig6_generalization.svg`.

| mode | params | in | ood5 | ood7 | ood9 |
|---|---|---|---|---|---|
| `sup`, with positions (phase 11) | 191,562 | 0.925 ± 0.009 | 0.670 ± 0.019 | 0.509 ± 0.035 | 0.435 ± 0.056 |
| `sup`, no positions | 191,690 | 0.968 ± 0.002 | 0.933 ± 0.004 | 0.922 ± 0.006 | 0.930 ± 0.013 |
| `curr`, with positions (phase 11) | 191,562 | 0.909 ± 0.017 | 0.647 ± 0.032 | 0.504 ± 0.030 | 0.459 ± 0.041 |
| `curr`, no positions | 191,690 | 0.965 ± 0.003 | 0.915 ± 0.014 | 0.908 ± 0.009 | 0.917 ± 0.012 |

The 128-parameter difference is the two extra operator embeddings (see the note at the end). `free`
mode was **not** rerun without positions.

## Phase 14 — Operators never used before

MODE (most frequent value) and RNG (max − min), fixed arity, with positions
(`results_newops.jsonl`).

| mode | in | ood5 | ood7 | ood9 |
|---|---|---|---|---|
| `sup` | 0.993 ± 0.001 | 0.947 ± 0.004 | 0.873 ± 0.026 | 0.811 ± 0.037 |
| `curr` | 0.995 ± 0.003 | 0.947 ± 0.002 | 0.896 ± 0.006 | 0.852 ± 0.010 |

"Never used" means these operators were absent from all earlier experiments; each model here is
trained and tested on MODE and RNG. This is not zero-shot transfer to unseen operators.

## Phase 15 — Variable arity: positions are not the cause, the lost count is

Plan: `plans/NIGHT4_PLAN.md`. Hidden size 192, 3 seeds.

| configuration (variable arity 2-7) | params | in (`sup`) | in (`curr`) | ood9 (`sup`) | ood9 (`curr`) |
|---|---|---|---|---|---|
| with positions (phase 12) | 147,466 | 0.611 ± 0.008 | 0.613 ± 0.003 | 0.347 ± 0.011 | 0.366 ± 0.003 |
| no positions (`results_vararity_nopos`) | 147,594 | 0.615 ± 0.007 | 0.613 ± 0.005 | 0.496 ± 0.012 | 0.493 ± 0.015 |
| no positions + count (`results_var_count`) | 160,522 | 0.895 ± 0.004 | 0.905 ± 0.011 | 0.800 ± 0.008 | 0.790 ± 0.015 |

Softmax attention returns a weighted average of the children. A sum of K values can be recovered from
their mean when K is constant, but not when K varies. Appending an embedding of the number of valid
children restores the accuracy, above the pre-registered threshold of 0.85. It remains about 0.06
below fixed arity (0.965), in a narrower model, with arity bounded at 7 and the empty slots visible as
PAD tokens.

## Phase 16 — One model, six operators

MAX, MIN, MED, SM, MODE, RNG mixed; fixed arity; no positions (`results_capstone.jsonl`).

| mode | params | in | ood5 | ood7 | ood9 |
|---|---|---|---|---|---|
| `sup` | 191,690 | 0.929 ± 0.004 | 0.891 ± 0.008 | 0.904 ± 0.005 | 0.920 ± 0.009 |
| `curr` | 191,690 | 0.933 ± 0.004 | 0.892 ± 0.011 | 0.902 ± 0.002 | 0.915 ± 0.002 |

## Phase 17 — Sensitivity to the curriculum schedule

Fraction of training spent in full teacher forcing before the linear weaning. Hard variant, fixed
arity, no positions, `curr` mode (`results_ch01.jsonl`, `results_nopos.jsonl`, `results_ch05.jsonl`).

| hold fraction | in | ood5 | ood7 | ood9 |
|---|---|---|---|---|
| 0.1 | 0.965 ± 0.004 | 0.917 ± 0.009 | 0.917 ± 0.008 | 0.927 ± 0.006 |
| 0.3 | 0.965 ± 0.003 | 0.915 ± 0.014 | 0.908 ± 0.009 | 0.917 ± 0.012 |
| 0.5 | 0.969 ± 0.004 | 0.930 ± 0.011 | 0.922 ± 0.009 | 0.930 ± 0.010 |

Only three values were tried; a hold of 0 (weaning from the first step) was not.

## What these results do not show

- **No external baseline.** No published latent-tree parser was trained on this task. The results
  support "the K-ary parser works here", not "it is better than existing methods". Published models
  reach 99%+ on standard ListOps without parse supervision.
- **A late and partial literature review** (`RELATED_WORK.md`). We do not claim that any component,
  or their combination, is new.
- **The hypernetwork ablation is confounded** by weight initialisation, which was left at its default.
- **One synthetic task.** Everything is ListOps or a variant of it, generated by
  `listops_experiment/listops_data.py`. This is **not** the ListOps benchmark of Long Range Arena:
  arity is fixed, trees are small, depth is controlled. The numbers are not comparable with published
  ListOps scores.
- **The best parser results use gold structure during training** (`sup` and `curr`). Without any
  structure signal the parser reaches 0.660, measured with positional embeddings only.
- **Small models, one GPU.** 0.15M to 1.1M parameters, single RTX 3070, 2-3 seeds.

## Index of data files

| file | phase | contents |
|---|---|---|
| `results_easy.jsonl` | 4 | R0, Etz1, Etz4 — four operators |
| `results_hard.jsonl` | 4 | R0, Etz1, Etz4 — hard variant |
| `results_ablation.jsonl` | 5 | Etz4_direct, Etz4_gated |
| `results_followup.jsonl` | 5 | R0 at matched budget, Etz1_direct |
| `results_histogram.jsonl` | 6 | Hist, Hist_big |
| `results_multistat.jsonl` | 7 | MultiStat at 3,000 iterations |
| `results_overnight.jsonl` | 8 | R0, R0_resid, MultiStat, MultiStat_resid, Linear, GRC at 10,000 iterations |
| `results_grc3k.jsonl` | 8 | GRC at 3,000 iterations |
| `results_saa.jsonl` | 9 | SAA_h2, SAA_h4, SAA_h8, SAA_hybrid |
| `results_parser.jsonl` | 10 | binary parser: l2r, soft |
| `results_gumbel.jsonl` | 10 | binary parser: gumbel |
| `results_sup_w02.jsonl`, `results_sup_w10.jsonl` | 10 | binary parser, structure-supervised, loss weight 0.2 and 1.0 |
| `results_consol.jsonl` | 10 | Etz4 and Hist_big at 10,000 iterations |
| `results_saaparser.jsonl` | 11 | SAA-Parser `sup` and `free` |
| `results_saap_vanilla.jsonl`, `results_saap_ms.jsonl` | 11 | composer ablation in `free` mode |
| `results_saap_curr.jsonl` | 11 | curriculum |
| `results_saap_allops.jsonl` | 11 | `free` mode, four operators |
| `results_vararity.jsonl` | 12 | variable arity, with positions |
| `results_nopos.jsonl` | 13 | fixed arity, no positions |
| `results_newops.jsonl` | 14 | MODE + RNG |
| `results_vararity_nopos.jsonl` | 15 | variable arity, no positions |
| `results_var_count.jsonl` | 15 | variable arity, no positions, with count |
| `results_capstone.jsonl` | 16 | six operators, no positions |
| `results_ch01.jsonl`, `results_ch05.jsonl` | 17 | curriculum hold 0.1 and 0.5 |

Each line is `{"seed": …, "kind": …, "params": …, "acc": {"in": …, "ood5": …, …}}`.

### Reproducing phases 4-12 exactly

The code now declares six operators; phases 4 to 12 were run when it declared four. The ids of MAX,
MIN, MED and SM did not change, but the embedding tables have two more rows. A model retrained today
with `--ops SM,MED` therefore has 128 more parameters than the one in the stored files (for example
191,690 instead of 191,562 for the SAA-Parser) and a different random initialisation at the same seed.
To reproduce those phases bit for bit, set `OPS = ["MAX", "MIN", "MED", "SM"]` in
`listops_experiment/listops_data.py`. We have not rerun the old phases under the new vocabulary, so
we cannot say how much the numbers would move.
