# Plan 5 — Three generalisation blocks (written before the experiment)

*English translation of the plan written before the runs (original: `docs/fr/plans/NIGHT3_PLAN.md`,
committed 2026-07-14 04:35; the results were committed 2026-07-15 18:44). Text in [brackets] was added
in translation. Results: `RESULTS.md`, phases 12 to 14.*

Three blocks chained automatically (each waits for the previous checkpoint).

## Block 1 — Variable arity

See `VARIABLE_ARITY_PLAN.md` (its reading grid is pre-registered there). sup+curr × 3 seeds, Kmin=2,
Kmax=7.

## Block 2 — The open problem: the OOD drop with depth (`--no_pos`)

**Mechanistic hypothesis (pre-registered).** Phase 11 leaves an open problem: even sup/curriculum lose
~50% accuracy at depth 9. Suspect: the **learned absolute positional embeddings** — OOD sequences are
*longer* [than the training ones], so the high positions are under-trained and inject noise. Yet the
SAA-Parser is **local** (a window scan): it may have no need for global positions at all.

**Test.** `listops_saa_parser.py --no_pos` (fixed arity, hard variant SM+MED, configuration identical
to phase 11), sup+curr × 3 seeds. Direct comparison with the phase 11 numbers (with positions):
sup 0.925 / 0.435 ood9; curr 0.909 / 0.459 ood9.

**Reading grid:**

| observation | conclusion |
|---|---|
| in ≈ same, ood9 clearly ↑ | **hypothesis supported** — absolute positions were what hurt OOD |
| in drops clearly | positions were needed for scoring (order information) — a clean negative |
| in ≈ same, ood9 ≈ same | positions are neutral; the OOD cause is elsewhere (to be documented) |

## Block 3 — Different operators: MODE and RNG

**Why these operators rather than boolean logic.** AND/OR/XOR over {0,1} are only special cases of
MIN/MAX/SM — a fake "new domain". MODE (most frequent value, ties → the smallest) and RNG (range
max−min) are **computations of a different kind**: frequency counting and combining two order
statistics. Neither reduces to the 4 original operators. Chance = 0.1 (10 classes), comparable with
the previous runs.

**Test.** `listops_saa_parser.py --ops MODE,RNG` (fixed arity), sup+curr × 3 seeds, standard
configuration.

**Reading grid:**

| observation | conclusion |
|---|---|
| sup ≥ ~0.85 and curr ≈ ~95%+ of sup | the mechanism (K-ary window + SAA + curriculum) **generalises to new semantics** |
| sup high but curr falls behind | the curriculum is sensitive to semantics — a real limit |
| sup low (< 0.7) | MODE/RNG exceed the composer's capacity — a limit of the composer, not of the parser |

## Comparability note

Adding MODE/RNG to the vocabulary (NUM_OPS 4→6) shifts the CLOSE/PAD ids and adds 2 embedding rows
(128 parameters) to ALL future trainings, block 2 included (191,690 vs 191,562 parameters in phase 11).
A negligible budget gap (+0.07%), noted here for transparency. [Block 1 had already run with the
4-operator vocabulary.]

[Outcome. Block 1: see VARIABLE_ARITY_PLAN.md. Block 2: first row, and in-dist accuracy rose too
(0.965-0.968). Block 3: first row (0.993 / 0.995).]
