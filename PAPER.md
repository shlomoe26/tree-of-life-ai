# The Tree of Life in deep learning: from a negative result on Kabbalah-inspired inductive biases to a K-ary latent parser trained by curriculum

*Etz HaChaim AI project — short report (workshop style). The original French version is in
`docs/fr/PAPER_FR.md`. Every number below can be recomputed from `listops_experiment/results_*.jsonl`;
see `RESULTS.md` for the full tables.*

## Abstract

Can the cosmology of Kabbalah (the Tree of Life: 4 Worlds, 10 Sefirot, 22 paths, the "Tzimtzum"
contraction) be turned into a competitive neural network architecture? We implement it with *Fractal
Hypernetworks* and compare it with standard baselines on two task families: character-level language
modelling, then evaluation of nested arithmetic expressions (ListOps). **Main result (negative):** the
Kabbalistic architecture consistently loses to a plain MLP, and the gap widens as the task gets harder.
**Mechanistic diagnosis (by ablation):** the culprit is not the cosmology but the *weight-generating
hypernetwork*; replacing it with direct weights recovers ~0.29 accuracy with 5× fewer parameters, and
its harm grows with depth. **First positive contribution:** once the rigid structure is dropped, the
baseline improves by *softly enriching* the aggregation, ending in a **learned set-attention
aggregator (SAA)** that raises accuracy to **0.988** (vs 0.959) when the tree is given. **Second
diagnosis, when trying to learn the structure itself** (from a flat sequence, with no gold tree):
three *binary* composition methods (greedy, Gumbel, supervised) all plateau at ~0.45, even with the
correct structure. In our setup the binary composers, not the search for structure, were the
bottleneck. (This is specific to our small models: published binary cells solve ListOps when the tree
is given, see §6.) **Second positive contribution, suggested by that observation:**
a **latent-structure parser that merges K-ary groups** with the SAA as composer, trained with a
**curriculum** (freeze the structure while the composer becomes competent, then wean it off), reaches
**0.909 ± 0.017** from a flat sequence — 98.3% of the ceiling obtained with the structure given
(0.925), far above any binary composition (~0.45) and above free parsing from scratch without
curriculum (0.660). An ablation adds a caveat: in free mode the quality of the composer (SAA vs
mean-pool) barely matters; the SAA's advantage appears only once the structure is correct. **Three
generalisation tests** then remove the remaining limits, each fix being pointed to by a diagnosis:
(i) the degradation with depth was an artefact of absolute positional embeddings — removing them
raises accuracy at depth 9 from 0.46 to **0.92** and even improves in-distribution accuracy (0.965);
(ii) under variable arity the ceiling drops to 0.61 because attention computes averages and loses the
*number* of operands — feeding that count back raises it to **0.90**; (iii) the mechanism transfers to
operators never seen before (0.99). **Final configuration:** a single 192k-parameter model, six mixed
operators, no positions, trained by curriculum on flat sequences, reaches **0.933 in-distribution and
0.915 at depth 9**. On this recursive computation: **rigid structured priors lose to soft,
high-dimensional features; and a K-ary latent structure can be learned by curriculum, entirely by
gradient, with no discrete search.** Two reservations frame these results: everything rests on a
synthetic task (ListOps), and we did not compare the parser with a published latent-tree method at
equal budget.

## 1. Introduction

The starting idea is radical: Kabbalah describes creation as information descending through 4
hierarchical Worlds, each made of 10 Sefirot (nodes) connected by 22 paths, with a contraction
(*Tzimtzum*) between levels. These motifs — a hierarchy of abstraction, a bottleneck, modularity,
parameter sharing — resemble good *inductive biases*. The scientific question: do these motifs, encoded
literally, yield a competitive architecture, or does the metaphor fail on contact with data?

Our contributions:
1. a **clean negative result** (multiple seeds, equal budget, pre-registered predictions) showing that
   the Kabbalistic architecture is dominated by a generic MLP;
2. a **mechanistic localisation** of the failure by ablation — the hypernetwork, not the cosmology;
3. a **positive design rule**: on this kind of task, progress comes from soft enrichment, not rigid
   structure — including against an architecture we believed optimal — ending in a learned
   set-attention aggregator (SAA, §4.6);
4. a **diagnosis** of why structure induction failed in our setup (our binary composers, not
   structure discovery, §4.7), which **points to** an architecture we had not tried — a K-ary latent-structure parser — validated
   by a **curriculum** that reaches 98.3% of a fully supervised ceiling, starting from a flat sequence
   and with no discrete search (§4.8);
5. three **generalisation tests** (§4.9), two of which end in a fix pointed to by a diagnosis: removing
   positional embeddings eliminates the degradation with depth, and feeding back the operand count
   restores computation under variable arity.

## 2. The Etz HaChaim architecture and the recursive "reversal"

**The fractal core.** Information passes through `num_worlds` successive Worlds. Each World is a graph
of 10 nodes (Sefirot) with a *learned* 10×10 adjacency matrix (so the "22 paths" are not fixed). The
transformation weights of each node are *generated* by a hypernetwork conditioned on position (world,
sefirah) — massive parameter sharing. Between two Worlds, a Tzimtzum layer compresses and re-expands
the representation (bottleneck). Switches allow ablations: 1 World without Tzimtzum (E0/Etz1) vs 4
full Worlds (E4/Etz4).

**The initial flaw.** In the "text" version, the input was copied identically onto the 10 Sefirot
(`x = x.repeat(1, 10, 1)`): the graph structure was never fed by the data. The Tree was, in effect, an
MLP in disguise.

**The reversal.** To bring in *real* structure, we turn the Etz into a **recursive cell applied at
each node of a syntax tree** (ListOps). At each internal node, the operator enters Sefirah S0, each
operand a distinct Sefirah (S1..S5), the remaining Sefirot serve as working registers, and the result
is read from the last Sefirah (Malchut). The same cell (shared weights) is applied bottom-up over the
whole tree, variable depth being absorbed by recursion. This is the first configuration in which
structure actually enters through the data rather than by copying.

## 3. Experimental protocol

**Task.** ListOps: nested expressions of operators (MAX, MIN, MED = median, SM = sum mod 10) over
digits 0–9; the output is one digit (10-class classification). Fixed arity K=5; controlled depth; tree
size capped (this isolates depth as the only variable of the extrapolation test and bounds memory). In
sections 4.1–4.6 the gold tree is given to every model: we therefore test the *combination cell*, not
structure discovery. That assumption is explicitly dropped in 4.7–4.9, where models receive a flat
sequence with no tree. Section 4.9 adds two operators (MODE = most frequent value, RNG = max − min) and
a variable-arity variant (2 to 7 operands per node).

**Safeguards (fixed before the runs).**
- *Calibration*: Etz models are run only if the baseline does not saturate in-distribution.
- *Judge no. 1 = extrapolation*: performance on trees deeper than those seen in training.
- *Equal budget*: all compared models have about the same number of parameters.
- *3 seeds* (exceptions listed in §6), mean ± standard deviation; pre-registered verdicts.

**Baseline (R0).** A "DeepSets" recursive cell: mean pooling of the children + operator embedding →
MLP. Permutation-invariant (the operators are symmetric), flexible, high-dimensional.

## 4. Results

### 4.1 Text tournament (Tiny Shakespeare, validation loss, lower is better)

| Bigram | Etz 4 worlds (1.1M) | Etz + attention | **Transformer (816k)** |
|---|---|---|---|
| 3.45 | 2.49 | 2.19 | **1.80** |

The Tree learns (it crushes the Bigram) but loses clearly to the Transformer; above all, going from 1
to 4 Worlds brings nothing (E0 ≈ E4). Diagnosis: a spatial graph applied to a sequence, with no real
input structure.

### 4.2 ListOps: structure enters through the data (accuracy, 3 seeds)

| variant | R0 (vanilla) | Etz 1 world | Etz 4 worlds |
|---|---|---|---|
| easy (4 ops) | **0.745** | 0.536 | 0.664 |
| hard (SM, MED) | **0.737** | 0.363 | 0.466 |

Two facts: (a) as soon as structure enters, **depth helps** (Etz4 ≫ Etz1) — the E0≈E4 curse of the
text setting is broken; (b) but the Etz stays **below** the flat MLP, and the gap **widens** on the
hard task (−0.08 → −0.27).

### 4.3 Autopsy: localising the failure (hard variant, equal budget)

| model | params | in-dist |
|---|---|---|
| Etz4 hyper | 1.1M | 0.466 |
| Etz4 + *gated* Tzimtzum | 1.1M | 0.504 |
| **Etz4 without hypernet (direct weights)** | **205k** | **0.752** |
| flat R0 | 208k | 0.780 |

**The hypernetwork is the culprit.** Replacing it with direct weights gives **+0.29** with **5× fewer
parameters**. The Tzimtzum contraction is only a minor factor (+0.04 as a learned gate). Moreover, the
hypernet's harm **grows with depth**: at 1 World, generated weights ≈ direct weights (≈0.36); at 4
Worlds the gap explodes. The shared generator becomes a bottleneck when it has to produce more
distinct weight sets. But at equal budget the flat MLP (0.780) is **≥** the corrected Etz (0.752):
once the hypernet is removed, the cosmological structure is neither better nor worse than a plain
DeepSets.

### 4.4 Which architecture the task calls for

We designed a *theoretically optimal* cell (the values are discrete, the operators symmetric): project
each child to a value, **sum = histogram**, read out through an operator-conditioned head — hence an
exact computation at any depth. **Prediction written in advance: it beats R0.** It failed (≈0.46, the
level of the original Etz): the 10-bin softmax is a hard, lossy bottleneck, worse than the soft 64-dim
mean-pool. By contrast, a *soft* enrichment of the aggregation — multi-statistic pooling
`[mean ; max ; min]` — **beats R0**: **0.791 vs 0.780** (3/3 seeds, equal budget), with a clearer gain
in extrapolation. `max`/`min` provide MAX/MIN for free while staying soft and high-dimensional.

### 4.5 The "ceiling" was under-training (correction)

Before attributing the ~0.79 ceiling to the cell, we checked the **training budget**: R0 and variants
retrained with **10,000 iterations + cosine schedule + 50,000 examples** (vs 3,000 / 15k), equal budget.

| model | in-dist @3000 | in-dist @10k+ | ood9 @10k+ |
|---|---|---|---|
| R0 | 0.780 | **0.951** | 0.884 |
| MultiStat | 0.791 | **0.959** | 0.911 |
| MultiStat + residual | — | **0.962** | 0.909 |
| Linear (linear composition) | — | 0.378 | 0.335 |
| GRC (binary fold) | 0.450 | 0.456 | 0.391 |

The MLP goes from 0.78 to **0.95**: the ceiling came from training, **not from the task** — which
corrects the interpretation of §4.4. The relative lessons still hold: MultiStat ≥ R0 (clear in OOD);
**linear composition fails** (MED and `mod` are non-linear); the **residual is neutral**; and the GRC
cell **stays at ~0.45 even at 10k** — its deficit is intrinsic to the cell (sequential fold ≠ symmetric
operators), not a lack of training.

### 4.6 A positive contribution: the learned aggregator (set attention)

Since enriching the aggregation *softly* helped (MultiStat), we pushed the idea: instead of
hand-picked statistics, a cell in which **H learned queries attend over the children** (in the manner
of PMA / Set Transformer) and extract H soft, permutation-invariant summaries. At equal budget, well
trained:

| model | in-dist | ood9 (depth 9) |
|---|---|---|
| R0 (mean) | 0.951 | 0.884 |
| MultiStat (fixed statistics) | 0.959 | 0.911 |
| **SAA (learned aggregate, 4 heads)** | **0.988** | **0.971** |

**Learned** aggregation clearly beats fixed statistics (+0.03 in-dist, **+0.06 OOD**), robustly (3
seeds, ±0.002). The gain is mostly **compositional** (extrapolation in depth), which suggests a real
learned composition rule. A notable sub-result: the *hybrid* variant (learned heads + explicit
max/min) is **worse** (0.970) than the purely learned one — once again, *a superfluous rigid component
hurts*.

### 4.7 Learning the structure itself: three failures, one unified diagnosis

The hardest frontier remained: receive a **flat sequence** (no tree) and learn the composition. Three
attempts, all by differentiable "easy-first" binary merging: (1) greedy straight-through: 0.469;
(2) + Gumbel and temperature annealing: 0.436; (3) + **supervision of the gold structure** (teacher
forcing): 0.411 in free parsing, **0.454 as the teacher-forced ceiling** (perfect structure supplied at
evaluation). Bounds: fixed left-to-right fold 0.420; GRC cell on the gold tree 0.456; SAA on the gold
tree **0.988**.

The pattern is unambiguous: supervision does teach the structure (free parsing ≈ teacher-forced
ceiling), but **the ceiling itself (~0.45) is the same for every binary composition** — fixed fold,
GRC, greedy, Gumbel, supervised. In these runs the bottleneck was not structure discovery but the
**binary composer**. Our working explanation at the time was that MED cannot be decomposed pairwise
(a median of medians is wrong), so a binary fold has to carry the whole multiset in its state, which
our composers did not learn. **This explanation does not hold as a general statement**: in the
literature, binary cells given the correct tree reach 98.7% (TreeLSTM, Nangia & Bowman 2018) and
99.95% (GRC, Ray Chowdhury & Caragea 2023) on ListOps, which includes the median and the modular sum.
What our data shows is narrower: at this size and training budget our binary composers failed where
K-ary aggregation (SAA) succeeded, and the failures of our GRC-style fold and of our latent parser
are the same failure. We do not know why our binary composers failed.

This diagnosis **points to an architecture we had not tried**: a latent-structure parser that merges
**K-ary groups** (operator + span of operands) through the learned aggregator, instead of pairs. A
hypothesis derived from our data ("chicken and egg"): unsupervised structure induction may have failed
*because* the binary composer could compute nothing — no signal rewarded a good merge; with a K-ary
composer that computes, the structure signal would become informative. Protocol pre-registered before
the experiment: `listops_experiment/plans/SAA_PARSER_PLAN.md`.

### 4.8 The SAA-Parser: the synthesis of the two frontiers, tested and validated

**Step 1 (testing the diagnosis).** On the new parser (window K+2, SAA composer), structure given
(`sup`): **0.925 ± 0.009** — against ~0.45 for our binary composers. Replacing the composer, with
everything else unchanged, removes the plateau, beyond the ≥0.9 threshold fixed in advance. In free
parsing from scratch: 0.660 ± 0.021 — well above our binary parsers, but far from the ceiling.

**Ablation (K-ary window vs composer).** Same parser, interchangeable cell, free mode: vanilla
(mean-pool) 0.634 ± 0.018, multistat 0.628 ± 0.003, SAA 0.660 ± 0.021 — **almost indistinguishable**.
Unlike the supervised regime (SAA ≫ binary), the quality of the composer does not show when the
structure has to be discovered: parsing errors dominate. The binary→K-ary jump comes first from the
**window itself** (grouping K operands at once), not specifically from learned attention.

**Curriculum (the headline result of this section).** Freeze the structure with teacher forcing for
the first 30% of training steps, then wean linearly towards free parsing: **0.909 ± 0.017**, i.e. 98.3%
of the supervised ceiling (0.925), against 0.660 for free-from-scratch and ~0.45 for any binary
composition (3 seeds, σ=0.017). This curriculum was designed after seeing the 0.660 of free mode; it was not in the pre-registered
plan (see §6). With that reservation, it clearly supports the chicken-and-egg hypothesis: letting the
composer become competent *before* entrusting it with structure discovery unlocks almost all of the
potential — a K-ary latent architecture, trained entirely by gradient (no discrete search), solves
the task here. (Latent-tree work often relies on discrete search such as beam search; we ran no such
baseline, see §6.) Bonus: the curriculum also improves OOD
(ood9 0.459 vs 0.278 for free) — better generalisation, not just a better in-dist score. Generality:
on full ListOps (4 operators, not only SM/MED), the free parser reaches 0.753 ± 0.007 — the mechanism
is not specific to the subset of hard operators used for the diagnosis.

**A limit that persisted — resolved in §4.9.** At this stage, the best modes lost ~45–50% accuracy
between in-dist and depth 9 (OOD).

### 4.9 Generalisation: positions, operators, arity

Three generalisation tests, then a combined configuration (pre-registered plans:
`VARIABLE_ARITY_PLAN.md`, `NIGHT3_PLAN.md`, `NIGHT4_PLAN.md` in `listops_experiment/plans/`; figure
`listops_experiment/figures/fig6_generalization.svg`).
All runs in this section are in `sup` or `curriculum` mode, 3 seeds:

**(a) Without positional embeddings — the open problem resolved.** Hypothesis: the window scan is
*local*; absolute positions, under-trained at OOD lengths, poison extrapolation. Configuration
identical to §4.8, only `--no_pos` changes: sup goes from 0.925/0.435 (in/ood9) to
**0.968 ± 0.002 / 0.930 ± 0.013**; curriculum from 0.909/0.459 to **0.965 / 0.917**. Removing positions
improves *even in-dist* accuracy and makes the degradation with depth almost vanish (only −0.04
between in-dist and depth 9). Yet another superfluous component that was hurting.

**(b) Operators never seen before (MODE = most frequent, RNG = max − min).** Semantics of a different
kind (counting, combining order statistics): sup/curriculum reach **0.993–0.995** in-dist, 0.81–0.85 at
depth 9. The mechanism is not specific to the operators of the diagnosis.

**(c) Variable arity [2,7] — first a limit, then a proven diagnosis.** The ceiling drops to ~0.61
*even with the gold structure* (in_tf = in), and a no_pos control shows that it is **not** positional
(0.615 ≈ 0.611). The diagnosis pointed to by the architecture: masked attention computes *weighted
averages*, whereas SM = sum requires the *number* of operands — at fixed K, sum = K × mean (a learnable
constant); at variable K, the count is destroyed by the softmax. **Feeding back an embedding of the
number of valid children** (+13k params) raises the ceiling from 0.61 to **0.90 ± 0.01** (0.80 at
depth 9). This is the project's third fix pointed to by a diagnosis (hypernet → direct weights; OOD →
no_pos; arity → count).

**(d) The combined configuration.** Everything together — a single 192k-parameter model, six mixed
operators (MAX, MIN, MED, SM, MODE, RNG), no positions, trained by curriculum on flat sequences:
**0.933 ± 0.004 in-dist, 0.915 ± 0.002 at depth 9** (almost flat generalisation, −0.018). The
curriculum matches the supervised ceiling there. Its robustness is checked: curr_hold ∈ {0.1; 0.3; 0.5}
gives 0.965–0.969 alike — what matters is the *progressive weaning*, not its setting.

## 5. Discussion

Across four independent structured architectures (cosmology; hypernet; "optimal" histogram; and even
the baseline before enrichment), a single pattern emerges: **rigid, discrete priors lose to soft,
high-dimensional features fed to a generic MLP.** The only gain above the baseline comes from a soft
enrichment, not from an imposed geometry. The most reusable mechanistic lesson concerns
**weight-generating hypernetworks**: they can cost *effective capacity* (the weights live on a
low-dimensional manifold), all the more as the number of requested weight sets grows — a cost hidden
behind the promise of "massive parameter sharing".

Epistemically, the project illustrates the caution required towards "elegant" architectures: two
predictions of victory based on elegance (the cosmology, then our histogram) turned out false; the
real improvements came from hypotheses *anchored in the data* and tested *without betting* — up to the
SAA-Parser, where the architecture itself was **pointed to** by a mechanistic diagnosis rather than
imagined. The ablation of §4.8 adds a nuance to the "soft beats rigid" thesis: when the structure is
*unknown*, the quality of the composer is not enough to show — it needs a curriculum that shields it
from exposure bias while it becomes competent. The pair "good composer + good curriculum" matters more
than the composer alone.

Section 4.9 repeats the same scheme three times: a failure is localised, the diagnosis points to a
specific fix, and the fix is tested against a threshold fixed in advance. Two of these fixes have
opposite signs. Removing positional embeddings is a *subtraction*: the window scan is local, and
absolute position information, poorly learned at lengths never seen, only did harm. Feeding back the
operand count is an *addition*: softmax-normalised attention is an average, and an average is not
enough to compute a sum once the number of terms varies. The rule "a superfluous component hurts"
therefore does not dispense with checking what the composer can actually represent.

## 6. Limitations

- **Published work already does better on the standard task.** A first literature pass, made after
  the experiments, is in `RELATED_WORK.md`. Its main points: (a) binary cells solve ListOps when the
  tree is given (98.7-99.95%), so our "binary plateau" reflects our composers and regime, not a
  property of binary composition; (b) unsupervised latent-tree models reach 99%+ on ListOps with no
  parse supervision (Havrylov et al. 2019; Ordered Memory; CRvNN; Beam Tree cells), far beyond our
  0.660 without structure signal; (c) that mean-like aggregation loses counts, and that several
  aggregators plus a degree term fix it, is known from graph networks (Xu et al. 2019; Corso et al.
  2020); (d) the harm we attribute to the hypernetwork may be an initialisation problem (Chang et al.
  2020), which we did not test.
- **No comparison with a published latent-tree method.** Our binary references (§4.7) are our own
  implementations; we did not train, at equal budget, a latent parser from the literature (for example
  a Gumbel-softmax Tree-LSTM). Without that comparison one can say the K-ary parser works, not that it
  is better than existing methods. None of the building blocks is new in isolation (latent trees, set
  attention, a gradual move from teacher forcing to free running); the contribution is their
  combination and the diagnosis that motivates it. We did not carry out a systematic literature
  review: we do not claim that K-ary latent parsing or this curriculum is new.
- **A single, synthetic task.** ListOps variants produced by our own generator (fixed arity, small
  trees, controlled depth). This is not the ListOps benchmark of Long Range Arena, and our numbers are
  not comparable with published ListOps scores. Nothing is tested on real data (code ASTs, molecules,
  language); the SAA-Parser makes that ground more plausible, not established.
- **Bounded, announced arity.** Variable arity (§4.9c) goes through a canonical window of size
  Kmax + 2 filled with PAD tokens: the model therefore sees where the empty slots are. With the count,
  the ceiling rises to 0.90, still 0.06 below the fixed-arity regime (0.965). These runs also use a
  narrower MLP (148–161k parameters against 192k), which rules out a strict comparison.
- **Free mode not re-tested without positions.** The 0.660 of free parsing without curriculum (§4.8)
  dates from the configuration with positions; the free/curriculum gap without positions is not
  measured.
- **Curriculum = structure supervision.** The curriculum uses the gold tree during training; only free
  mode does without it. "No discrete search" does not mean "no supervision".
- **The curriculum was not pre-registered.** The plan written before the experiment
  (`SAA_PARSER_PLAN.md`) tested the chicken-and-egg hypothesis through free mode, with a threshold of
  0.85; the result (0.660) falls in the "structure partially learned" row. The curriculum was designed
  after seeing that number. It supports the hypothesis, but it is a post-hoc experiment and should be
  read as one. The tests of §4.9 (positions, operators, count, robustness) do have thresholds written
  before the runs.
- **Seed counts and variance.** 3 seeds everywhere except: Linear and GRC at 10k, the fixed l2r fold,
  and the 10k retraining of Etz4 and the histogram cell (2 seeds each). The ± values are population
  standard deviations over those seeds, so they are only indicators of spread. Two results are noisy:
  Etz4 without hypernet (0.752 ± 0.069) and the greedy binary parser (0.469 ± 0.054).
- **Text tournament (§4.1): a single run** of 1,000 iterations per model, without multiple seeds, and
  at unequal budgets (Transformer 816k, Etz 1.1M). It is a starting observation, not a controlled
  measurement.
- **The MultiStat gain is modest** (+0.008 in-dist at 10k, +0.027 at depth 9).
- Etz4 and the histogram cell were retrained at 10k iterations (0.469 and 0.517, 2 seeds): their
  deficit does not come from the training budget.
- **Small scale** (models of ~0.15–1.1M parameters, one RTX 3070). The phenomena could differ at large
  scale.
- Two earlier limits are removed in §4.9: the degradation with depth (an artefact of positional
  embeddings; ood9 goes from 0.46 to 0.92) and the specificity to the SM/MED operators.

## 7. Conclusion

Kabbalah inspires good *principles* (hierarchy helps when the data is structured), but does not, as
it stands, produce a competitive *architecture*: the generating hypernetwork penalises it, and once
corrected, its cosmological structure is neutral against an MLP. Pursuing the question "which
architecture does this task really call for", two positive contributions emerge, each pointed to by
the diagnosis of the previous one rather than by elegance: (1) on a given tree, the **learned
set-attention aggregator (SAA)** reaches 0.988 in-dist / 0.971 OOD at equal budget; (2) on a **flat
sequence**, the finding that our *binary* composers plateau at ~0.45 even with the correct structure
points to a **K-ary latent-structure parser**, which, trained with a
**curriculum** that shields the composer from exposure bias, reaches **0.909 — 98.3% of the ceiling
obtained with the structure given**, entirely by gradient, with no discrete search. The generalisation
tests then correct its two measured weaknesses: without positional embeddings the degradation with
depth disappears (0.965 in-distribution, 0.917 at depth 9); with the operand count, variable arity
rises from 0.61 to 0.90. The combined configuration — six operators, one 192k-parameter model — reaches
**0.933 / 0.915**. This is an honest negative result on Kabbalah, mechanistically diagnosed, that
leads to two positive architectures and one rule of method: *look for what the diagnosis points to,
not for what is elegant — and when a capable component fails in an unsupervised regime, suspect the
order of learning before the architecture itself.* What this work does not show: that the K-ary
parser does better than a latent parser from the literature (published models are far ahead on
standard ListOps, see `RELATED_WORK.md`), and that it holds outside a synthetic task.

---

*Code and raw data (one JSON line per run): `listops_experiment/`. Full tables and the index of data
files: `RESULTS.md`. Plans written before the runs: `listops_experiment/plans/` (English translations;
the French originals are in `docs/fr/plans/`). Figures: `listops_experiment/figures/`, regenerated
from the data by `listops_experiment/make_figures.py`.*
