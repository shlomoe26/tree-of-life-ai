"""
The REVERSAL: the Etz is no longer the network that reads the whole expression, it is the
CELL applied at each node of the syntax tree. Structure enters through the data
(S0=operator, S1..SK=distinct children) instead of through repeat(1,10,1).

Cells defined here (all share the signature cell(op_emb, children) -> (out, aux_loss)):
- VanillaCell (R0)  : generic recursive cell (DeepSets, mean pooling).
- MultiStatCell     : R0 with [mean ; max ; min] pooling.
- SetAggCell (SAA)  : learned set-attention aggregator; optional child mask and child count.
- LinearCell, ResidualCell, GRCFoldCell, HistogramCell : diagnostic cells.
- EtzCell           : reuses EtzChaimFractalAI UNCHANGED. The num_worlds/tzimtzum switches
                      give the ablation E0 (1 world) vs E4 (4 worlds + Tzimtzum).

Evaluation is batched by height bucket (no in-place autograd tricks: node vectors are kept
in a Python list and stacked to gather the children).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from etz_hachaim.research.fractal_worlds import EtzChaimFractalAI
from listops_data import NUM_OPS, NUM_DIGITS  # noqa: F401 (NUM_OPS is used elsewhere)


class VanillaCell(nn.Module):
    """R0. Mean pooling (permutation-invariant: MAX/MIN/MED/SM are symmetric), then an MLP.
    `hidden` is sized to match the parameter budget of the model it is compared with."""
    def __init__(self, d, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, op_emb, children):           # (n,d), (n,K,d)
        pooled = self.norm(children.mean(dim=1))   # (n,d)
        out = self.mlp(torch.cat([op_emb, pooled], dim=-1))
        return out, op_emb.new_zeros(())


class MultiStatCell(nn.Module):
    """Enriched R0: instead of mean-pool alone, concatenate [mean ; max ; min] of the children.
    max/min give MAX/MIN for free (order statistics that mean-pool cannot extract), and the
    distributional information helps MED/SM, while staying SOFT and high-dimensional (no hard
    bottleneck, unlike the histogram cell, which failed)."""
    def __init__(self, d, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(3 * d)
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, op_emb, children):               # (n,d), (n,K,d)
        pooled = torch.cat([children.mean(dim=1),
                            children.max(dim=1).values,
                            children.min(dim=1).values], dim=-1)   # (n, 3d)
        out = self.mlp(torch.cat([op_emb, self.norm(pooled)], dim=-1))
        return out, op_emb.new_zeros(())


class SetAggCell(nn.Module):
    """SAA: aggregator LEARNED by attention over a set (in the manner of PMA / Set Transformer).
    H learned queries ('seeds') attend over the children and extract H soft summaries that are
    permutation-invariant (softmax over the children). It generalises MultiStat: the mean is
    uniform attention, the max is roughly peaked attention; here the model LEARNS which
    statistics to extract instead of having them fixed by hand.
    hybrid=True also concatenates explicit [max ; min].
    child_mask (forward) masks absent children; count_feat=True appends an embedding of the
    number of valid children (needed under variable arity, see below)."""
    def __init__(self, d, n_heads=4, hidden=256, hybrid=False, count_feat=False, max_count=16):
        super().__init__()
        self.n_heads, self.hybrid, self.count_feat = n_heads, hybrid, count_feat
        self.seeds = nn.Parameter(torch.randn(n_heads, d) * 0.02)   # H learned queries
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        agg_dim = n_heads * d + (2 * d if hybrid else 0) + (d if count_feat else 0)
        # count_feat: masked attention computes AVERAGES; SM (a sum) needs the COUNT of valid
        # operands, which the softmax destroys when arity varies -> it is fed back explicitly.
        if count_feat:
            self.count_emb = nn.Embedding(max_count + 1, d)
        self.norm = nn.LayerNorm(agg_dim)
        self.mlp = nn.Sequential(
            nn.Linear(agg_dim + d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, op_emb, children, child_mask=None):   # (n,d), (n,K,d), (n,K) bool or None
        k = self.to_k(children)                             # (n,K,d)
        v = self.to_v(children)                             # (n,K,d)
        scores = torch.einsum('hd,nkd->nhk', self.seeds, k) / (k.shape[-1] ** 0.5)  # (n,H,K)
        if child_mask is not None:                          # EXPLICIT masking of PAD children
            scores = scores.masked_fill(~child_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)               # permutation-invariant (softmax over K)
        summ = torch.einsum('nhk,nkd->nhd', attn, v).reshape(children.shape[0], -1)  # (n, H*d)
        if self.hybrid:
            summ = torch.cat([summ, children.max(dim=1).values, children.min(dim=1).values], dim=-1)
        if self.count_feat:
            n_valid = (child_mask.sum(1) if child_mask is not None
                       else torch.full((children.shape[0],), children.shape[1], device=children.device))
            summ = torch.cat([summ, self.count_emb(n_valid.long().clamp(0, self.count_emb.num_embeddings - 1))], dim=-1)
        return self.mlp(torch.cat([op_emb, self.norm(summ)], dim=-1)), op_emb.new_zeros(())


class LinearCell(nn.Module):
    """LINEAR composition: no non-linearity in the combination (a single matmul).
    Motivated by length-generalisation theory (a linear relation helps compositional
    generalisation). SM is roughly linear (a sum), MED is not: a diagnostic test.
    Small by nature (one matmul)."""
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(2 * d, d)            # [mean(children) ; op] -> output, no activation

    def forward(self, op_emb, children):
        out = self.lin(torch.cat([children.mean(dim=1), op_emb], dim=-1))
        return out, op_emb.new_zeros(())


class ResidualCell(nn.Module):
    """Residual connection against error compounding: the children's information passes through.
    out = cell(op, children) + sigmoid(alpha) * mean(children). alpha is learned (init 0 -> half pass)."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, op_emb, children):
        out, intent = self.inner(op_emb, children)
        return out + torch.sigmoid(self.alpha) * children.mean(dim=1), intent


class GRCFoldCell(nn.Module):
    """GRC-style cell (Gated Recursive Cell): a gated, soft, high-dimensional BINARY combiner.
    On a K-ary node it FOLDS left to right: h = op_emb ; h = GRC(h, child_k) for each k.
    This is the standard binarisation of ListOps (the sequence [op, c1..cK] composed pairwise).
    NB: a GRC-style approximation (sigmoid gates + tanh candidate + LayerNorm), NOT a faithful
    reproduction of the published cell. The sequential fold is NOT permutation-invariant,
    which is the point of the test."""
    def __init__(self, d, hidden):
        super().__init__()
        self.d = d
        self.proj = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, 4 * d))
        self.norm = nn.LayerNorm(d)

    def _combine(self, a, b):                              # (n,d),(n,d) -> (n,d)
        i, fl, fr, c = self.proj(torch.cat([a, b], dim=-1)).chunk(4, dim=-1)
        h = torch.sigmoid(i) * torch.tanh(c) + torch.sigmoid(fl) * a + torch.sigmoid(fr) * b
        return self.norm(h)

    def forward(self, op_emb, children):                  # (n,d), (n,K,d)
        h = op_emb
        for k in range(children.shape[1]):
            h = self._combine(h, children[:, k, :])
        return h, op_emb.new_zeros(())


class EtzCell(nn.Module):
    """Kabbalistic cell. 10 Sefirot: S0=operator, S1..SK=children, the rest are learned
    'scratch' registers (NOT a copy of the input). The result is read from Malchut (slot 9).
    Reuses EtzChaimFractalAI as is."""
    def __init__(self, d, K, num_worlds, tzimtzum, intent_weight=0.1, weight_mode='hyper', tzimtzum_mode='fixed'):
        super().__init__()
        assert 1 + K <= 9, "the Malchut slot (9) must stay free for the output"
        self.d, self.K, self.intent_weight = d, K, intent_weight
        self.core = EtzChaimFractalAI(d_model=d, num_worlds=num_worlds, tzimtzum_active=tzimtzum,
                                      weight_mode=weight_mode, tzimtzum_mode=tzimtzum_mode)
        self.num_scratch = 10 - 1 - K              # remaining slots -> working registers
        self.scratch = nn.Parameter(torch.randn(self.num_scratch, d) * 0.02)

    def forward(self, op_emb, children):           # (n,d), (n,K,d)
        n = op_emb.shape[0]
        scratch = self.scratch.unsqueeze(0).expand(n, -1, -1)          # (n, num_scratch, d)
        x_nodes = torch.cat([op_emb.unsqueeze(1), children, scratch], dim=1)  # (n,10,d)
        intent = x_nodes.clone()                   # "Keter": the original intent (as in the language model)
        x_out, intent_loss = self.core(x_nodes, intent)
        out = x_out[:, 9, :]                        # Malchut
        return out, self.intent_weight * intent_loss


class HistogramCell(nn.Module):
    """The cell we predicted would win (it did not): no cosmology, no hypernet.
    Each child is projected to a distribution over the 10 values; the SUM over children forms
    a histogram (how many are 0, 1, ... 9); an operator-conditioned head reads the result.
    MAX/MIN/MED/SM are all exact functions of that histogram, so in theory the computation
    can be exact at any depth. In practice the 10-bin softmax is a hard, lossy bottleneck."""
    def __init__(self, d, hidden=256, n_values=NUM_DIGITS):
        super().__init__()
        self.value_proj = nn.Linear(d, n_values)           # child (d) -> value logits (10)
        self.readout = nn.Sequential(                      # [histogram(10) ; op(d)] -> output (d)
            nn.Linear(n_values + d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, op_emb, children):                   # (n,d), (n,K,d)
        val_dist = torch.softmax(self.value_proj(children), dim=-1)   # (n,K,10) soft one-hot
        histogram = val_dist.sum(dim=1)                              # (n,10) counts per value
        out = self.readout(torch.cat([histogram, op_emb], dim=-1))   # (n,d)
        return out, op_emb.new_zeros(())


class RecursiveTreeModel(nn.Module):
    """Embeddings (leaves + operators) -> cell applied recursively -> readout head on the
    root -> logits over 10 digits."""
    def __init__(self, d, cell):
        super().__init__()
        self.d = d
        self.leaf_emb = nn.Embedding(NUM_DIGITS, d)
        self.op_emb = nn.Embedding(NUM_OPS, d)
        self.cell = cell
        self.head = nn.Linear(d, NUM_DIGITS)

    def evaluate(self, trees, device):
        # Post-order flattening: children before parents.
        digit, opid, children, height, roots = [], [], [], [], []

        def visit(n):
            if n.is_leaf:
                gid = len(height)
                digit.append(n.digit); opid.append(-1); children.append(None); height.append(0)
                return gid
            cids = [visit(c) for c in n.children]
            gid = len(height)
            digit.append(-1); opid.append(n.op_id); children.append(cids); height.append(n.height)
            return gid

        for t in trees:
            roots.append(visit(t))
        N = len(height)

        # buckets by height
        buckets = {}
        for g in range(N):
            buckets.setdefault(height[g], []).append(g)

        rows = [None] * N

        # leaves (height 0) in one go
        leaf_ids = buckets.get(0, [])
        if leaf_ids:
            ldig = torch.tensor([digit[g] for g in leaf_ids], device=device, dtype=torch.long)
            lemb = self.leaf_emb(ldig)
            for i, g in enumerate(leaf_ids):
                rows[g] = lemb[i]

        total_intent = torch.zeros((), device=device)
        Hmax = max(height) if height else 0
        for h in range(1, Hmax + 1):
            ids = buckets.get(h, [])
            if not ids:
                continue
            ops = torch.tensor([opid[g] for g in ids], device=device, dtype=torch.long)
            oemb = self.op_emb(ops)                                            # (n,d)
            cvec = torch.stack([torch.stack([rows[c] for c in children[g]], 0) for g in ids], 0)  # (n,K,d)
            out, intent = self.cell(oemb, cvec)
            total_intent = total_intent + intent
            for i, g in enumerate(ids):
                rows[g] = out[i]

        root_vec = torch.stack([rows[g] for g in roots], 0)                    # (B,d)
        return root_vec, total_intent

    def forward(self, trees, device):
        root_vec, intent = self.evaluate(trees, device)
        return self.head(root_vec), intent


# ---- budget helpers ----------------------------------------------------------

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def find_vanilla_hidden(target_cell_params, d):
    """Smallest `hidden` such that VanillaCell(d,hidden) >= target (equal budget)."""
    best = 16
    for hidden in range(16, 4097, 8):
        if count_params(VanillaCell(d, hidden)) >= target_cell_params:
            return hidden
        best = hidden
    return best


def find_hidden(cell_builder, target_cell_params):
    """Generic: smallest hidden such that cell_builder(hidden) >= target (equal budget)."""
    best = 16
    for hidden in range(16, 4097, 8):
        if count_params(cell_builder(hidden)) >= target_cell_params:
            return hidden
        best = hidden
    return best
