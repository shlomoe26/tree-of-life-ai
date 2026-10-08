"""
BINARY latent parser: learn the tree STRUCTURE softly (no gold tree, no discrete beam).

The model receives a flat sequence of ListOps tokens and has to discover how to compose it,
through a straight-through 'easy-first' merge (in the spirit of Gumbel Tree-LSTM, Choi et al.
2018; this is our own small implementation, not a reproduction of that model):
  at each step every adjacent pair is scored, a straight-through softmax picks one (forward =
  merge the best pair, backward = soft gradient), the pair is merged by a gated binary
  combiner, and the sequence gets one token shorter.
Sequence reconstruction is fully differentiable:
  R'[k] = a_k*R[k] + y[k]*C[k] + b_k*R[k+1]   (a_k = P(merge after k), b_k = P(merge before k)).

Kinds:
  l2r    : fixed left-to-right fold (no learned structure), the lower bound
  soft   : greedy straight-through easy-first parser
  gumbel : same, with Gumbel noise at training time and temperature annealing
  sup    : teacher forcing of the gold binary merges (structure supervision)
Reference upper bound: SAA on the gold tree = 0.988 (measured with listops_run.py).

All four plateau around 0.42-0.47 on the hard variant; see RESULTS.md, phase 10.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import argparse, random, time, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from listops_data import make_dataset, NUM_DIGITS, NUM_OPS  # the tree generator is reused

# --- Sequence vocabulary: digits 0-9 | operators | CLOSE ---
# (16 and 17 with the current six operators; 14 and 15 when phases 10-12 were run with four)
CLOSE = NUM_DIGITS + NUM_OPS
VOCAB = NUM_DIGITS + NUM_OPS + 1


def serialize(node):
    """Tree -> token sequence in prefix notation: op, children..., CLOSE."""
    if node.is_leaf:
        return [node.digit]
    toks = [NUM_DIGITS + node.op_id]
    for c in node.children:
        toks += serialize(c)
    toks.append(CLOSE)
    return toks


def gold_merge_positions(seq):
    """Left BINARY merge order that rebuilds the gold tree, as positions in the shrinking
    sequence. Used for structure supervision (teacher forcing). Length = len(seq)-1."""
    kinds = ['close' if t == CLOSE else ('op' if t >= NUM_DIGITS else 'val') for t in seq]
    pos = []
    while len(kinds) > 1:
        c = kinds.index('close')                 # leftmost complete node
        q = c - 1
        while kinds[q] != 'op':
            q -= 1
        for _ in range(c - q):                   # binary fold of op+operands+close, all at position q
            pos.append(q)
            kinds[q] = 'val'
            del kinds[q + 1]
    return pos


def make_seq_dataset(n, depths, K, p_deep, seed, max_nodes, op_ids):
    trees, _ = make_dataset(n, depths, K, p_deep, seed=seed, max_nodes=max_nodes, op_ids=op_ids)
    out = []
    for t in trees:
        s = serialize(t)
        out.append((s, t.value, gold_merge_positions(s)))
    return out


class BinaryComposer(nn.Module):
    """Gated binary combiner (soft, high-dimensional): (a,b) -> parent."""
    def __init__(self, d, hidden):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, 4 * d))
        self.norm = nn.LayerNorm(d)

    def forward(self, a, b):                                   # (m,d),(m,d) -> (m,d)
        i, fl, fr, c = self.proj(torch.cat([a, b], dim=-1)).chunk(4, dim=-1)
        return self.norm(torch.sigmoid(i) * torch.tanh(c) + torch.sigmoid(fl) * a + torch.sigmoid(fr) * b)


class ParserModel(nn.Module):
    """BATCHED version: right padding + a mask of the real prefix (merges keep the real tokens
    contiguous at the start). About 20x faster than a per-example loop."""
    def __init__(self, d=64, hidden=256, mode="soft", max_len=256):
        super().__init__()
        self.d, self.mode = d, mode
        self.tok_emb = nn.Embedding(VOCAB, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.composer = BinaryComposer(d, hidden)
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.head = nn.Linear(d, NUM_DIGITS)

    def _pack(self, batch_seqs, device):
        B = len(batch_seqs)
        Lmax = max(len(s) for s in batch_seqs)
        idx = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        for i, s in enumerate(batch_seqs):
            idx[i, :len(s)] = torch.tensor(s, device=device, dtype=torch.long)
        lengths = torch.tensor([len(s) for s in batch_seqs], device=device)
        pos = torch.arange(Lmax, device=device)
        R = self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0)      # (B,Lmax,d)
        mask = pos.unsqueeze(0) < lengths.unsqueeze(1)             # (B,Lmax)
        return R, mask, lengths

    def forward(self, batch_seqs, device, tau=1.0, gold_positions=None):
        R, mask, lengths = self._pack(batch_seqs, device)
        B, Lmax, d = R.shape
        zero = R.new_zeros(())
        if self.mode == "l2r":
            h = R[:, 0]
            for k in range(1, Lmax):
                cand = self.composer(h, R[:, k])
                m = mask[:, k].unsqueeze(1).float()
                h = m * cand + (1 - m) * h
            return self.head(h), zero
        # teacher forcing (mode 'sup'): gold positions as a (B, Lmax-1) tensor, -1 = no step
        gp = None
        if self.mode == "sup" and gold_positions is not None:
            gp = torch.full((B, Lmax - 1), -1, dtype=torch.long, device=device)
            for i, g in enumerate(gold_positions):
                if g: gp[i, :len(g)] = torch.tensor(g, device=device)
        struct_loss, struct_n = zero, 0
        for step in range(Lmax - 1):
            a, b = R[:, :-1], R[:, 1:]
            C = self.composer(a.reshape(-1, d), b.reshape(-1, d)).reshape(B, -1, d)
            cur = C.shape[1]                                       # number of pairs = current length - 1
            pair_valid = mask[:, :-1] & mask[:, 1:]
            scores = (C @ self.query).masked_fill(~pair_valid, -1e9)
            no_valid = pair_valid.sum(1) == 0
            if no_valid.any():
                scores[no_valid, -1] = 0.0
            if self.mode == "sup" and gp is not None:              # teacher forcing
                tgt = gp[:, step]                                  # (B,)
                valid = tgt >= 0
                # finished examples (tgt=-1): no-op merge in the padding (last pair),
                # NOT at position 0 (which would destroy the root of short sequences)
                tgt_c = torch.where(valid, torch.clamp(tgt, 0, cur - 1),
                                    torch.full_like(tgt, cur - 1))
                if valid.any():
                    struct_loss = struct_loss + F.cross_entropy(scores[valid], tgt_c[valid], reduction='sum')
                    struct_n += int(valid.sum())
                y_hard = torch.zeros(B, cur, device=device).scatter_(1, tgt_c.unsqueeze(1), 1.0)
                soft = torch.softmax(scores, dim=1)
                y = y_hard + (soft - soft.detach())
            else:
                logit = scores
                if self.mode == "gumbel" and self.training:
                    u = torch.rand_like(scores).clamp_(1e-9, 1 - 1e-9)
                    logit = scores + (-torch.log(-torch.log(u)))
                logit = logit / tau
                soft = torch.softmax(logit, dim=1)
                y_hard = torch.zeros_like(soft).scatter_(1, logit.argmax(1, keepdim=True), 1.0)
                y = y_hard + (soft - soft.detach())
            csum = torch.cumsum(y, dim=1)
            a_k, b_k = 1.0 - csum, csum - y
            R = a_k.unsqueeze(2) * R[:, :-1] + y.unsqueeze(2) * C + b_k.unsqueeze(2) * R[:, 1:]
            lengths = torch.clamp(lengths - 1, min=1)
            mask = torch.arange(R.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
        struct_loss = struct_loss / max(1, struct_n)
        return self.head(R[:, 0]), struct_loss


@torch.no_grad()
def accuracy(model, data, device, chunk=128, teacher_force=False):
    model.eval(); correct = total = 0
    for i in range(0, len(data), chunk):
        b = data[i:i + chunk]
        gp = [g for _, _, g in b] if teacher_force else None   # tf = gold structure; otherwise FREE parsing
        logits, _ = model([s for s, _, _ in b], device, gold_positions=gp)
        pred = logits.argmax(-1)
        gold = torch.tensor([v for _, v, _ in b], device=device)
        correct += (pred == gold).sum().item(); total += len(b)
    model.train(); return correct / total


def train(model, train_data, eval_sets, device, iters, batch, lr, log_every, seed, cosine, warmup,
          tau_start=2.0, tau_end=0.5, struct_w=1.0):
    model.to(device).train()
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = None
    if cosine:
        def lam(it):
            if it < warmup: return (it + 1) / max(1, warmup)
            p = (it - warmup) / max(1, iters - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    t0 = time.time()
    for it in range(1, iters + 1):
        tau = tau_start + (tau_end - tau_start) * (it / iters)     # linear annealing 2.0 -> 0.5
        b = [train_data[rng.randrange(len(train_data))] for _ in range(batch)]
        seqs = [s for s, _, _ in b]
        gold = torch.tensor([v for _, v, _ in b], device=device)
        gp = [g for _, _, g in b] if model.mode == "sup" else None
        logits, struct = model(seqs, device, tau=tau, gold_positions=gp)
        loss = F.cross_entropy(logits, gold) + struct_w * struct   # struct=0 outside 'sup' mode
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # against the gradient explosion we observed
        opt.step()
        if sched: sched.step()
        if it % log_every == 0 or it == iters:
            accs = {k: accuracy(model, ts, device) for k, ts in eval_sets.items()}
            if model.mode == "sup":                            # teacher-forced ceiling (gold structure)
                accs["in_tf"] = accuracy(model, eval_sets["in"], device, teacher_force=True)
            print(f"    it {it:5d} loss {loss.item():.3f} tau {tau:.2f} | " +
                  " ".join(f"{k} {v:.3f}" for k, v in accs.items()), flush=True)
    print(f"    [done {time.time()-t0:.0f}s]", flush=True)
    res = {k: accuracy(model, ts, device) for k, ts in eval_sets.items()}
    if model.mode == "sup":
        res["in_tf"] = accuracy(model, eval_sets["in"], device, teacher_force=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="soft,l2r")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--p_deep", type=float, default=0.2)
    ap.add_argument("--ops", default="SM,MED")
    ap.add_argument("--train_depths", default="1,2,3,4")
    ap.add_argument("--ood_depths", default="5,7,9")
    ap.add_argument("--max_nodes", type=int, default=120)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log_every", type=int, default=2500)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data_seed", type=int, default=1234)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--struct_w", type=float, default=1.0, help="weight of the structure loss (sup mode)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_parser.jsonl"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from listops_data import OP_TO_ID as OPMAP
    op_ids = [OPMAP[o.strip()] for o in args.ops.split(",")] if args.ops else None
    tds = [int(x) for x in args.train_depths.split(",")]
    ods = [int(x) for x in args.ood_depths.split(",")]
    mn = args.max_nodes

    train_data = make_seq_dataset(args.n_train, tds, args.K, args.p_deep, args.data_seed, mn, op_ids)
    eval_sets = {"in": make_seq_dataset(args.n_test, tds, args.K, args.p_deep, args.data_seed + 1, mn, op_ids)}
    for d in ods:
        eval_sets[f"ood{d}"] = make_seq_dataset(args.n_test, [d], args.K, args.p_deep, args.data_seed + 100 + d, mn, op_ids)

    lens = [len(s) for s, _, _ in train_data]
    print("=" * 64)
    print(f"Binary latent parser | device={device} | ops={args.ops}")
    print(f"train={len(train_data)} | seq length min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")
    print(f"Reference upper bound (SAA on the gold tree) = 0.988")
    print("=" * 64, flush=True)

    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            l = l.strip()
            if l:
                r = json.loads(l); done[(r["seed"], r["kind"])] = r["acc"]

    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"\n#### SEED {seed} ####", flush=True)
        for kind in [k.strip() for k in args.kinds.split(",")]:
            if (seed, kind) in done:
                print(f"[resume] {kind} seed {seed} -> skip", flush=True); continue
            torch.manual_seed(seed)
            model = ParserModel(d=args.d, hidden=args.hidden, mode=kind)
            npar = sum(p.numel() for p in model.parameters())
            print(f"\n--- {kind} | params={npar:,} | seed={seed} ---", flush=True)
            res = train(model, train_data, eval_sets, device, args.iters, args.batch,
                        args.lr, args.log_every, seed, args.cosine, args.warmup,
                        struct_w=args.struct_w)
            with open(args.out, "a") as f:
                f.write(json.dumps({"seed": seed, "kind": kind, "params": npar, "acc": res}) + "\n")
            del model
            if device == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
