# Plan 4 — Generalisation test: VARIABLE arity (written before the experiment)

*English translation of the plan written before the runs (original:
`docs/fr/plans/VARIABLE_ARITY_PLAN.md`, committed 2026-07-09 17:31; the results were committed
2026-07-15 18:44). Text in [brackets] was added in translation. Results: `RESULTS.md`, phases 12 and 15.*

*Follows SAA_PARSER_PLAN.md. It addresses a limitation of the fixed-arity parser: "fixed K+2 window,
not tested with variable arity".*

## 1. The technical problem

The SAA-Parser scans windows of **fixed** size `K+2` (operator + K operands + CLOSE). Generalising to
an arity that varies per node breaks the differentiable trick used to rebuild the sequence (the shift
`R' = a·R[:-(K+1)] + y·C + b·R[K+1:]` assumes that a constant length is removed at each step).
Rebuilding that trick for a removal of *variable length per example of the batch* is a non-trivial
alignment problem (each example would pick a window of a different length at the same step, leaving
residual sequences of different sizes).

## 2. The chosen solution: canonical arity Kmax with masked padding

- **Generator**: *genuinely* variable arity per node, drawn in `[Kmin=2, Kmax=7]`.
- **Serialisation**: each group is *completed* to exactly `Kmax` operands with a dedicated **PAD**
  token (a new vocabulary token), so that each internal node always takes `Kmax+2` raw tokens. This
  restores a fixed-size window. It is the standard padding of deep learning (like a batch of sequences
  of different lengths): the model has to compute while ignoring the empty slots.
  [Consequence, stated in the paper's limitations: the empty slots are visible in the input.]
- **Composer (SAA)**: `SetAggCell` receives a validity mask of the children (a new parameter) and sets
  the attention logits of PAD children to `-inf` before the softmax — **explicit** masking, not a hope
  that the network learns to ignore an embedding.
- **Window scanner / scorer**: unchanged in form (window `Kmax+2`).

This is an accepted compromise: we do not test *unbounded* arity, but an arity that is *variable and
unknown in advance, up to a ceiling*.

## 3. Implementation (files)

- `listops_data.py`: `gen_tree_var(depth, Kmin, Kmax, ...)` — arity drawn per node.
- `listops_model.py`: `SetAggCell.forward` accepts an optional `child_mask` (True = valid).
- `listops_saa_parser_vararity.py`: padded serialisation; gold reduction order that treats PAD as
  "already resolved"; the child mask is passed to the composer.
- **Unit test BEFORE any long run** (a padding bug had already been caught this way twice): check
  that (a) a short sequence gives the same result alone and batched with a long one; (b) the PAD mask
  is applied. [Only (a) is actually asserted in the code; (b) is a smoke check that the masked forward
  pass runs.]

## 4. Protocol (identical to phase 11 for direct comparability)

Hard variant (SM, MED), Kmin=2, Kmax=7, same node cap and depths, budget ~200k (adjusted for the
padding), 10k iterations + cosine + 50k examples, 3 seeds. Modes: `sup` (gold structure) and `curr`
(curriculum, hold 30% then weaning). [The runs used hidden=192, i.e. 147k parameters, less than the
192k of the fixed-arity runs.]

## 5. Reading grid (pre-registered)

| observation | conclusion |
|---|---|
| sup ≥ ~0.85 | the K-ary composer + masking handles variable arity; the diagnosis transfers |
| sup drops clearly (< 0.7) | the padding/masking regime introduces a difficulty of its own, not just arity |
| curriculum ≈ phase 11 ratio (~98% of its own sup ceiling) | **the curriculum generalises** — not an artefact of fixed K=5 |
| curriculum ≪ phase 11 ratio | the curriculum is sensitive to the regularity of arity — a real limit to document |

## 6. Honest calibration

Padding to Kmax is not a *complete* generalisation to unbounded arity. If this test succeeds, the
honest conclusion is "the mechanism generalises to bounded variable arity, with explicit masking" —
not "to arbitrary arity".

[Outcome: sup = 0.611 (second row) and curriculum = 0.613 (third row: 100% of its own ceiling). The
count feature that later raised this to 0.90 is the subject of NIGHT4_PLAN.md.]
