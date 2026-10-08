# Plan 6 — Count feature, combined configuration, curriculum robustness (written before the experiment)

*English translation of the plan written before the runs (original: `docs/fr/plans/NIGHT4_PLAN.md`,
committed 2026-07-16 02:05; the results were committed 2026-07-16 15:47). Text in [brackets] was added
in translation. Results: `RESULTS.md`, phases 15 to 17.*

Chained automatically. A control run was in progress when this was written:
`results_vararity_nopos.jsonl` (does variable arity drop because of positions?). Early signal at that
time [2 of 6 runs finished]: no (~0.60 ≈ 0.61 with positions); the limit looked real.

## Block A — The count feature (the last limit, attacked mechanistically)

**Hypothesis pointed to by the architecture**: masked attention computes *weighted averages*; yet
SM = sum requires the *number* of operands. At fixed K, sum = K × mean (a learnable constant); at
variable K, the count is destroyed by the softmax → the composer cannot compute SM.
**Fix**: feed an embedding of the number of valid children back into the composer (`--count`).
Configuration: variable arity, no_pos + count, sup+curr × 3 seeds → `results_var_count.jsonl`.

| observation | conclusion |
|---|---|
| sup ≥ ~0.85 | diagnosis **proven**: the variable-arity limit was the lost count → lifted |
| sup ~0.7-0.85 | the count is part of the explanation, not all of it |
| sup ≈ 0.61 unchanged | diagnosis refuted — the difficulty is elsewhere (to be documented) |

## Block B — Combined configuration: ONE model, SIX operators, flat sequences

Everything learned so far, combined: fixed K=5 parser + no_pos + curriculum, on the mix
MAX, MIN, MED, SM, MODE, RNG. sup+curr × 3 seeds → `results_capstone.jsonl`. Expected (if the gains
compose): ≥0.9 in-dist, strong OOD thanks to no_pos. No bet — we measure.

## Block C — Robustness of the curriculum (is 30% special?)

curr_hold = 0.1 and 0.5 (vs 0.3 used everywhere), fixed K=5 parser, SM+MED, no_pos, curr × 3 seeds
→ `results_ch01.jsonl`, `results_ch05.jsonl`.
If the three values give about the same → the curriculum is robust (not a fragile hyperparameter).
If 0.1 collapses → the composer really must be left to learn first.

[Outcome. Block A: sup = 0.895, first row. Block B: 0.929 (sup) / 0.933 (curr) in-dist, 0.920 / 0.915
at depth 9. Block C: 0.965 / 0.965 / 0.969 for hold = 0.1 / 0.3 / 0.5.]
