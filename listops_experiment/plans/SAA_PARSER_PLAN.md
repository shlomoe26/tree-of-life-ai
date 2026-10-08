# Plan 3 — SAA-Parser: design plan (written BEFORE any experiment)

*English translation of the plan written before the runs (original: `docs/fr/plans/SAA_PARSER_PLAN.md`,
committed 2026-07-07 14:08; the results were committed 2026-07-09 04:22). Text in [brackets] was added
in translation. Results: `RESULTS.md`, phase 11.*

## 1. Where this architecture comes from (the reasoning, not the elegance)

Two facts established by our experiments:

1. **Learned aggregation (succeeded)**: **learned K-ary** aggregation (SAA, set attention) computes
   our operators at 0.988 on the gold tree — when every binary cell plateaus at ~0.45.
2. **Structure learning (diagnosed)**: structure induction fails **not** because the structure cannot
   be learned (when supervised, free parsing reaches its ceiling: 0.411 ≈ 0.454), but because **every
   binary composition hits the same wall at ~0.42-0.46** — MED cannot be decomposed pairwise, so a
   binary fold would have to carry the whole multiset.

**The synthesis this points to**: a latent-structure parser that merges **whole K-ary groups** (an
operator + its span of operands) in one step, with the **SAA as composer**. To our knowledge, within
what we had looked at, latent-tree parsers are binary. [This was written without a literature review;
it is not a claim of novelty.]

## 2. The central hypothesis ("chicken and egg"), pre-registered

Unsupervised structure induction may have failed *because* the binary composer could compute nothing
(ceiling 0.45): good merge or bad merge, the loss barely moved → **no signal** to learn the structure.
With a K-ary composer that *can* compute (0.99), a good merge pays off immediately → the structure
signal becomes informative → **the two halves bootstrap each other**.

## 3. The architecture

A flat sequence of tokens (prefix notation: `op c1 … cK ]`). Reduction loop:

1. **Span scorer**: for each position i, a score s_i = "is the window [i .. i+K+1] a reducible group
   (op, K operands, CLOSE) whose operands are already leaves/values?". (K is fixed = 5 in our
   generator → a fixed window of size K+2; an announced limitation, like the generator's arity.)
2. **Straight-through selection** (ST softmax as in the binary parser — the selection mechanism was
   NOT the problem, we keep it).
3. **K-ary composition**: the selected group is reduced to ONE node by the **SAA** (op → operator
   embedding, the K operands → children of the SetAggCell). The CLOSE token is consumed. The sequence
   gets K+1 tokens shorter.
4. Repeat until 1 node remains → classification head.

Differentiable reconstruction of the sequence: the same cumulative-masking principle as the binary
parser (generalised to removing a block of K+1 positions).

## 4. Protocol (identical to the previous runs → direct comparisons)

Hard variant (SM, MED), K=5, cap of 120 nodes, training depths 1-4 / OOD 5-7-9, 50k examples, 10k
iterations + cosine, batch 64, 3 seeds, data_seed 1234, budget ~200-250k parameters.

Two steps, in order:
- **Step 1 — validating the diagnosis (supervised structure)**: teacher forcing of the gold
  reductions. If the ceiling `in_tf` jumps beyond 0.45 (towards ~0.9+), the diagnosis "binarisation
  was the wall" is **proven causally**. If in_tf stays at ~0.45, our diagnosis is wrong (and we will
  have to say so).
- **Step 2 — the real test (free structure)**: unsupervised parsing. This is the test of the
  chicken-and-egg hypothesis.

## 5. Reading grid (pre-registered, not to be moved)

| observation | conclusion |
|---|---|
| Step 1: in_tf ≥ ~0.9 | binarisation diagnosis **proven**; continue |
| Step 1: in_tf ~0.45 | diagnosis **refuted** — the wall is elsewhere; stop and document it |
| Step 2: free ≥ ~0.85 (≈ in_tf) | **chicken-and-egg hypothesis validated**: a K-ary latent parser with learned aggregation beats every binary composition |
| Step 2: free ≫ 0.47 but < in_tf | structure partially learned: a real gain, incomplete co-learning |
| Step 2: free ≈ 0.45-0.47 | chicken-and-egg refuted: even with a good composer, no signal — co-learning is the next wall |

Every outcome is worth reporting; the last two close the question cleanly.

## 6. Risks identified in advance

- The fixed K+2 window is a strong simplification (accepted, like the generator's arity).
- The scorer must learn "my operands are leaves" — if it errs early, errors cascade (the same exposure
  bias as the binary parser). Step 1 neutralises it (teacher forcing), step 2 measures it.
- Beware of padding/batch bugs (lesson from the supervised binary parser: the short-roots bug — check
  with a unit test BEFORE any long run).

## 7. Honest calibration (repeated so as not to fool ourselves)

Even in case of full success: a result **shown on a niche task**. To "count" beyond it, it would have
to transfer (other tasks, variable arity, the standard ListOps benchmark).

[Outcome. Step 1: in_tf = 0.925, first row. Note that "binarisation diagnosis proven" overstates what
this shows: published binary cells solve ListOps on gold trees, so the plateau belongs to our binary
composers, not to binary composition (see `RELATED_WORK.md`). Step 2: free = 0.660, which is the **fourth** row
("structure partially learned"), not the third. The curriculum that later reached 0.909 was designed
after seeing this number; it is not part of this pre-registered plan.]
