# Plan 7 — Two follow-up experiments after the literature check (written before the runs)

Written 2026-10-08, in English, directly in the public repository. Motivation: `RELATED_WORK.md`
found two places where our conclusions are either contradicted by published work or confounded.
These experiments try to find out what actually happened in our runs.

Nothing below has been run yet, except the two measurements explicitly marked as such.

## Experiment A — Why did our binary composers plateau at 0.45?

**What we know.** On the hard variant (SM + MED, arity 5), with the gold tree, our GRC-style binary
fold reaches 0.456 at 10,000 iterations (2 seeds) and 0.450 at 3,000. Published binary cells on gold
trees reach 98.7% (TreeLSTM) to 99.95% (GRC) on standard ListOps. So binary composition can do it;
ours did not.

**Candidate explanations, and the run that tests each.**

| # | hypothesis | test |
|---|---|---|
| A1 | The cell cannot even compute one operator application | train and test on **depth 1 only** (one operator over 5 digits), report accuracy **per operator** |
| A2 | One operator is learned and the other is not | the per-operator split of A1 and A4 |
| A3 | The 64-d state is too small to carry five values through a fold | same runs at d = 128, same parameter budget |
| A4 | Our GRC-style cell normalises its state at every step and has no additive memory path | a second binary cell, `LSTMFold`: Tree-LSTM-style fold with a memory cell |
| A5 | Not enough training | GRC at 30,000 iterations |

Two measurements already made while timing the code (400 iterations, depth 1, one seed, not a
result): GRC 0.528 and LSTMFold 0.524 in distribution. A second smoke test of the per-operator
code (150 iterations, depth 1, one seed) printed MED 0.83 / SM 0.08 for GRC and MED 0.86 / SM 0.11
for R0. These numbers were seen before this plan was committed, so hypothesis A2 is not blind: there
is already a hint that the operator our folds miss is the **modular sum, not the median** — the
opposite of the explanation given in the paper. At 150 iterations the MLP control misses it too, so
this says nothing yet about where each model ends up.

**Runs.** Equal budget (~208k parameters, `--budget_ref Etz4_direct`), cosine schedule, gradient
clipping at 1.0 for the binary cells (the stored runs had none; this is a deviation, chosen because
folds are recurrent). Controls R0 and SAA_h4 are run in the same conditions.

- A-d1: depth 1 only, 5,000 iterations, 20,000 examples, d = 64: GRC, LSTMFold, R0, SAA_h4 × 3 seeds.
- A-d1-wide: same, d = 128: GRC, LSTMFold × 3 seeds.
- A-full: depths 1-4, OOD 5/7/9, 10,000 iterations, 50,000 examples: LSTMFold d = 64 × 2 seeds,
  GRC d = 128 × 2 seeds, LSTMFold d = 128 × 2 seeds.
- A-long: GRC d = 64, 30,000 iterations × 1 seed.

**Reading grid.**

| observation | conclusion |
|---|---|
| depth 1: binary cells ≥ 0.9 on both operators | they can compute one application; the failure is in recursion over depth (errors compound, or the representation handed to the parent is poor) |
| depth 1: one operator near 1.0, the other near chance (0.1-0.3) | a specific operator is not learned by our pairwise folds in this regime; name it |
| depth 1: both operators partial, similar | an optimisation or capacity problem of the fold itself |
| any full-task binary configuration ≥ 0.9 | the plateau was that factor (width, cell design or training length), not binary composition |
| full task: every binary configuration ≤ 0.55 | still unexplained at our scale; we report the gap with the literature as open |
| controls: R0 and SAA_h4 at depth 1 should be ≥ 0.95 | otherwise the depth-1 setup itself is broken and A-d1 is void |

## Experiment B — Is the hypernetwork deficit an initialisation effect?

**What we know.** Etz with 4 worlds reaches 0.466 with generated weights and 0.752 with direct
weights (3,000 iterations, 3 seeds). The hypernetwork used PyTorch's default initialisation.
Chang, Flokas & Lipson (2020) show that this puts generated weights at the wrong scale.

**Measurement already made (not a training result).** At initialisation the generated weight
matrices have standard deviation 0.198-0.207 over three seeds, against 0.125 (= 64^-0.5) for the
direct ablation: 1.6× too large in standard deviation, 2.6× in variance. The mismatch exists but is
moderate, and the inputs are layer-normalised, so we do not expect it to explain a 0.29 gap. That
expectation is written here so that it can be wrong.

**Intervention.** `Etz4_hinit`: rescale the hypernetwork's output layer at initialisation so that
the generated weights start at standard deviation 64^-0.5. This is an empirical calibration to the
same target as hyperfan-in, not the analytic formula of the paper. Nothing else changes; the
parameter count is identical.

**Runs.** Hard variant, 3,000 iterations, 15,000 examples, no cosine (the regime of the original
ablation): Etz4 (control) × 2 seeds, Etz4_hinit × 3 seeds, Etz4_direct (control) × 1 seed.

**Reading grid.**

| observation | conclusion |
|---|---|
| controls far from 0.47 (Etz4) or 0.75 (direct) | the regime was not reproduced; B is void until it is |
| Etz4_hinit ≥ 0.70 | the deficit was mostly initialisation; the "hypernetwork costs capacity" reading is withdrawn |
| Etz4_hinit between 0.55 and 0.70 | initialisation is part of it |
| Etz4_hinit ≤ 0.55 | scale at initialisation is not the cause; other explanations (shared low-dimensional weight manifold, optimisation through the generator) remain untested |

## Limits known in advance

- Few seeds (1 to 3) and long runs: this is a diagnostic, not a benchmark.
- Experiment A uses our generator and our budget; a positive result would show that a binary cell
  *can* learn our variant, not reproduce any published number.
- Experiment B tests one aspect of initialisation (output scale). A negative result does not rule
  out other initialisation or optimisation effects.
- The operator vocabulary now has six entries; models have 128 more parameters than the stored
  phase 5 and phase 8 runs. The controls are there to absorb that difference.

Outputs: `results_binary_depth1.jsonl`, `results_binary_depth1_d128.jsonl`,
`results_binary_full.jsonl` (d = 64), `results_binary_full_d128.jsonl`, `results_binary_long.jsonl`,
`results_hypernet_init.jsonl`.
