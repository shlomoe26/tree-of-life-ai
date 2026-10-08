# Related work, and what it changes

This note was written after the experiments, on 2026-10-08. The project was carried out without a
literature review; this is a first, non-exhaustive pass, done to check the claims in `PAPER.md`
against published work. It changes how several of them should be read.

Sources marked **[read]** were opened and checked for the facts quoted here (through the arXiv /
ar5iv pages, not always the final published version). Sources marked **[cited]** are standard
references given from general knowledge of the field; their details were not re-verified for this
note.

## 1. The most important correction: binary composition is not a wall on ListOps

`PAPER.md` §4.7 reports that every binary composer we tried plateaus near 0.45 on our hard variant,
even with the correct structure, and reads this as "binarisation is the bottleneck: a median cannot
be decomposed pairwise".

Published results do not support that as a general statement:

- **Nangia & Bowman (2018), ListOps [read].** ListOps contains MAX, MIN, MED and SUM_MOD. A binary
  TreeLSTM given the ground-truth parses reaches **98.7%** (128-d). The reference parses are
  left-branching inside each list, i.e. the list is folded pairwise.
- **Ray Chowdhury & Caragea (2023), Beam Tree Recursive Cells [read].** A binary gated recursive cell
  (GRC) on gold trees reaches **99.95%** near-IID on ListOps and 99.9-100% on longer sequences.

So binary cells *can* carry what a median or a modular sum needs when the tree is given. Our
GRC-style fold on the gold tree reached 0.456. The honest reading is therefore narrower than the one
in the paper: **our small binary composers, in our training regime, did not learn these operators**.
What remains true is the within-project comparison: at the same budget and with the same training
recipe, the K-ary window with a set-attention composer learned the task and our binary composers did
not.

**Follow-up experiment (added 2026-10-09; `RESULTS.md`, phase 18).** We split accuracy by operator.
Our binary folds learn the **median perfectly** and leave the **modular sum at chance**: the paper's
explanation named the wrong operator. On a task reduced to one operator application, the same folds
reach 100% on the sum when trained six times longer with a doubled batch: they were under-trained,
not incapable. State width (64 vs 128) and cell design (GRC-style vs Tree-LSTM-style) made no
difference. On the full task, however, 30,000 iterations left the sum at chance, so the gap with the
published numbers is only partly explained.

## 2. Unsupervised structure learning on ListOps was already solved

- **Havrylov, Kruszewski & Joulin (2019), Cooperative Learning of Disjoint Syntax and Semantics
  [read].** A binary parser trained with REINFORCE (self-critical baseline, PPO) and a separate
  Tree-LSTM composer reach **99.2 ± 0.5%** on ListOps **with no parse supervision**, against 57.6%
  for Gumbel Tree-LSTM and 60.7% for RL-SPINN. They attribute earlier failures to *co-adaptation*:
  the composer learns faster than the parser, which locks the parser into early strategies.
- **Shen et al. (2019), Ordered Memory [cited]**; **Ray Chowdhury & Caragea (2021), CRvNN [read,
  abstract]**; **Beam Tree Recursive Cells (2023) [read]**. In the Beam Tree paper's ListOps table,
  models that induce structure without gold trees reach 99.4-99.9% near-IID (Ordered Memory 99.88,
  CRvNN 99.82, BT-GRC + OneSoft 99.92).
- **Csordás, Irie & Schmidhuber (2022), Neural Data Router [read, abstract]** reports near-perfect
  accuracy on a ListOps variant testing generalisation across computational depth, with a modified
  Transformer.

Compared with these, our parser is far behind where it matters: without any structure signal it
reaches 0.660, and its good results (0.91-0.97) use gold structure during training. The tasks are
not identical (see §6), so the numbers cannot be put side by side, but there is no sense in which
this repository improves on published latent-tree methods.

Our "chicken and egg" hypothesis (the composer must compute before the parser can learn) is a cousin
of the co-adaptation analysis of Havrylov et al., who separate and re-pace the two modules instead of
supervising the structure first.

## 3. K-ary composition

We did not find a paper that scores and merges whole latent K-ary groups with a set-attention
composer, but we cannot conclude it does not exist.

- **Tai, Socher & Manning (2015), Tree-LSTM [cited]** includes a Child-Sum variant that aggregates
  any number of children of a *given* tree.
- **Dyer et al. (2016), Recurrent Neural Network Grammars [cited]** reduce n-ary constituents in a
  supervised shift-reduce parser.
- **Ray Chowdhury & Caragea (2023), Recursion in Recursion [read]** uses a k-ary balanced tree as
  an outer recursion. That k-ary structure is fixed (consecutive chunks of k), not induced; a latent
  binary tree is induced inside each chunk.

## 4. Learned set aggregation, and why a count is needed

- **Zaheer et al. (2017), Deep Sets [cited]**: sum-then-MLP as a universal form for set functions.
  Our baseline R0 is this with a mean.
- **Lee et al. (2019), Set Transformer [cited]**: pooling by multi-head attention with learned seed
  vectors. Our SAA cell is a small version of this.
- **Xu, Hu, Leskovec & Jegelka (2019), How Powerful are Graph Neural Networks? [read, via search
  summary]**: sum aggregation can be injective over multisets; **mean aggregation is not**, it
  captures proportions and loses counts.
- **Corso et al. (2020), Principal Neighbourhood Aggregation [read, via search summary]**: combine
  several aggregators (mean, max, min, std) and add *degree scalers*, which let a mean recover a sum.

Our two "findings" on aggregation are therefore known results in another guise: `[mean ; max ; min]`
pooling is a subset of PNA's aggregators, and feeding the number of valid children back to an
attention (mean-like) aggregator is what degree scalers do.

## 5. Positions, curricula, hypernetworks

- **Positional information and length generalisation.** **Kazemnejad et al. (2023) [read, via search
  summary]** find that decoder-only Transformers with no positional encoding generalise to longer
  inputs better than common explicit encodings on their tasks. This is consistent with our
  observation that removing learned absolute positions closed the depth gap. It is debated: a 2024
  formal analysis argues the opposite for learnable absolute encodings under a different setup. Our
  case is also simpler than theirs: our absolute embeddings for long positions were barely trained.
- **From teacher forcing to free running.** **Bengio et al. (2015), Scheduled Sampling [cited]**
  anneals the probability of feeding ground truth during training. Our curriculum is the same idea
  applied to structure decisions. We did not find it applied to latent tree structure in this pass.
- **Hypernetworks.** **Ha, Dai & Le (2017) [cited]** introduced them. **Chang, Flokas & Lipson
  (2020), Principled Weight Initialization for Hypernetworks [read, via search summary]** show that
  standard initialisations give generated weights at the wrong scale and hurt training. Our
  hypernetwork used a default initialisation. Our conclusion that "the hypernetwork costs effective
  capacity" was therefore confounded.
  **Follow-up experiment (added 2026-10-09; `RESULTS.md`, phase 19).** Generated weights start at
  1.6 times the standard deviation of the direct ablation. Rescaling them to the same scale changes
  nothing (0.464 against 0.463; direct weights 0.646). Output scale at initialisation is not the
  cause. The per-operator split suggests another reading, untested: with direct weights the model
  starts to learn the modular sum within the budget, with the hypernetwork it does not.

## 6. Our task is not the published ListOps

Published ListOps (Nangia & Bowman) has variable numbers of arguments, mean depth about 9.6, 90k
training examples, and sequences up to hundreds or thousands of tokens in later work. Ours comes from
a custom generator: arity fixed at 5 (or 2-7 in one experiment), at most 120 nodes, depths 1-4 in
training. Our "depth 9" test is easy by those standards. Generalisation to more arguments than seen
in training ("argument generalisation") is a known hard axis in the literature (for example 63.7% at
15 arguments for BT-GRC, 75.05% for Ordered Memory in the Beam Tree paper [read]); our variable-arity
experiment does not test it, since training and test arities are the same.

## 7. What this leaves of the project's claims

| claim in the write-up | status after this pass |
|---|---|
| The Etz architecture loses to a flat MLP | stands (internal comparison) |
| The hypernetwork is the cause | the ablation stands and was reproduced; initialisation scale is ruled out; "it costs capacity" is still an interpretation |
| Learned set attention beats fixed pooling when the tree is given | stands; it is an instance of known set-pooling results |
| "Binarisation is the wall" | **withdrawn**. The plateau is the modular sum left unlearned; at depth 1 more training fixes it, on the full task 30,000 iterations did not |
| K-ary parser + curriculum reaches 0.91-0.97 | stands as measured; uses gold structure; far from published unsupervised results |
| Removing positions fixes depth generalisation | stands; consistent with prior work |
| A count feature fixes variable arity | stands; a known property of mean-like aggregation |

## References

- Bengio, Vinyals, Jaitly, Shazeer. *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.* NeurIPS 2015.
- Chang, Flokas, Lipson. *Principled Weight Initialization for Hypernetworks.* ICLR 2020. arXiv:2312.08399.
- Choi, Yoo, Lee. *Learning to Compose Task-Specific Tree Structures.* AAAI 2018.
- Corso, Cavalleri, Beaini, Liò, Veličković. *Principal Neighbourhood Aggregation for Graph Nets.* NeurIPS 2020. arXiv:2004.05718.
- Csordás, Irie, Schmidhuber. *The Neural Data Router.* ICLR 2022. arXiv:2110.07732.
- Dyer, Kuncoro, Ballesteros, Smith. *Recurrent Neural Network Grammars.* NAACL 2016.
- Ha, Dai, Le. *HyperNetworks.* ICLR 2017.
- Havrylov, Kruszewski, Joulin. *Cooperative Learning of Disjoint Syntax and Semantics.* NAACL 2019. arXiv:1902.09393.
- Kazemnejad, Padhi, Ramamurthy, Das, Reddy. *The Impact of Positional Encoding on Length Generalization in Transformers.* NeurIPS 2023. arXiv:2305.19466.
- Lee, Lee, Kim, Kosiorek, Choi, Teh. *Set Transformer.* ICML 2019.
- Nangia, Bowman. *ListOps: A Diagnostic Dataset for Latent Tree Learning.* NAACL Student Research Workshop 2018. arXiv:1804.06028.
- Ray Chowdhury, Caragea. *Modeling Hierarchical Structures with Continuous Recursive Neural Networks.* ICML 2021. arXiv:2106.06038.
- Ray Chowdhury, Caragea. *Beam Tree Recursive Cells.* 2023. arXiv:2305.19999.
- Ray Chowdhury, Caragea. *Efficient Beam Tree Recursion.* NeurIPS 2023. arXiv:2307.10779.
- Ray Chowdhury, Caragea. *Recursion in Recursion: Two-Level Nested Recursion for Length Generalization with Scalability.* 2023. arXiv:2311.04449.
- Shen, Tan, Hosseini, Lin, Sordoni, Courville. *Ordered Memory.* NeurIPS 2019. arXiv:1910.13466.
- Tai, Socher, Manning. *Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks.* ACL 2015.
- Tay et al. *Long Range Arena: A Benchmark for Efficient Transformers.* ICLR 2021.
- Xu, Hu, Leskovec, Jegelka. *How Powerful are Graph Neural Networks?* ICLR 2019. arXiv:1810.00826.
- Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov, Smola. *Deep Sets.* NeurIPS 2017.
