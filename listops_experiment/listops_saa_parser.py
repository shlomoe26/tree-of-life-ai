"""
SAA-Parser: a latent-structure parser that merges whole K-ARY GROUPS
(op + K operands + CLOSE, a fixed window of K+2 tokens) with the SAA as composer.

It combines two earlier findings (see plans/SAA_PARSER_PLAN.md, written before any run):
- the SAA (learned K-ary aggregation) computes at 0.988 when the gold tree is given;
- every BINARY composition plateaus at ~0.45 (MED cannot be decomposed pairwise), however
  the structure is chosen (l2r / GRC / soft / gumbel / supervised).

Modes:
  sup  = teacher forcing of the gold reductions (+ cross-entropy on the window scorer);
         evaluated both free and teacher-forced (in_tf).
  free = unsupervised parsing (straight-through softmax), no structure signal at all.
  curr = curriculum: teacher forcing with probability 1 for the first `curr_hold` fraction
         of training, then the probability decays linearly to 0. NOTE: it uses the gold
         structure during training; only `free` does without it.

Reported "in"/"ood*" accuracies are always FREE parsing; "in_tf" is teacher-forced.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import argparse, random, time, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from listops_data import make_dataset, NUM_DIGITS, NUM_OPS
from listops_parser import serialize, CLOSE, VOCAB
from listops_model import SetAggCell, VanillaCell, MultiStatCell


def gold_group_positions(seq, K):
    """Gold order of the K-ary reductions: at each step, the leftmost reducible group
    (op followed by K values then CLOSE), as a position in the shrinking sequence."""
    kinds = ['c' if t == CLOSE else ('o' if t >= NUM_DIGITS else 'v') for t in seq]
    pos = []
    while len(kinds) > 1:
        p = -1
        for i in range(len(kinds) - K - 1):
            if kinds[i] == 'o' and all(k == 'v' for k in kinds[i + 1:i + 1 + K]) and kinds[i + K + 1] == 'c':
                p = i; break
        assert p >= 0, "no reducible group (malformed sequence?)"
        pos.append(p)
        kinds[p:p + K + 2] = ['v']
    return pos


class SAAParserModel(nn.Module):
    def __init__(self, d=64, hidden=256, K=5, mode="free", max_len=256, cell="saa", use_pos=True):
        super().__init__()
        self.d, self.K, self.mode = d, K, mode
        self.tok_emb = nn.Embedding(VOCAB, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.use_pos = use_pos   # OOD ablation: window parsing is local, does it need positions at all?
        # ablation "what carries the gain": K-ary window alone (vanilla/multistat) vs with the SAA
        if cell == "saa":
            self.cell = SetAggCell(d, n_heads=4, hidden=hidden)
        elif cell == "multistat":
            self.cell = MultiStatCell(d, hidden)
        elif cell == "vanilla":
            self.cell = VanillaCell(d, hidden)
        else:
            raise ValueError(cell)
        self.query = nn.Parameter(torch.randn(d) * 0.02)      # window scorer
        self.head = nn.Linear(d, NUM_DIGITS)

    def _pack(self, seqs, device):
        B = len(seqs); Lmax = max(len(s) for s in seqs)
        idx = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            idx[i, :len(s)] = torch.tensor(s, device=device, dtype=torch.long)
        lengths = torch.tensor([len(s) for s in seqs], device=device)
        pos = torch.arange(Lmax, device=device)
        R = self.tok_emb(idx) + (self.pos_emb(pos).unsqueeze(0) if self.use_pos else 0)
        mask = pos.unsqueeze(0) < lengths.unsqueeze(1)
        return R, mask, lengths

    def forward(self, seqs, device, gold_positions=None):
        K = self.K
        R, mask, lengths = self._pack(seqs, device)
        B, Lmax, d = R.shape
        zero = R.new_zeros(())
        steps = (Lmax - 1) // (K + 1)
        gp = None
        if self.mode in ("sup", "curr") and gold_positions is not None:
            gp = torch.full((B, steps), -1, dtype=torch.long, device=device)
            for i, g in enumerate(gold_positions):
                if g: gp[i, :len(g)] = torch.tensor(g, device=device)
        struct_loss, struct_n = zero, 0
        for step in range(steps):
            cur = R.shape[1]
            W = cur - (K + 1)                                   # number of possible (K+2) windows
            if W <= 0: break
            # K-ary candidates in parallel: window j -> SAA(op=R[j], children=R[j+1..j+K])
            ops = R[:, :W]                                      # (B,W,d)
            children = R.unfold(1, K, 1)[:, 1:W + 1]            # (B,W,d,K) sliding windows
            children = children.permute(0, 1, 3, 2).reshape(B * W, K, d)
            C, _ = self.cell(ops.reshape(B * W, d), children)
            C = C.reshape(B, W, d)
            # validity: the window lies entirely inside the real sequence
            win_valid = mask.unfold(1, K + 2, 1)[:, :W].all(dim=2)   # (B,W)
            scores = (C @ self.query).masked_fill(~win_valid, -1e9)
            no_valid = win_valid.sum(1) == 0                    # examples already reduced to one token
            if no_valid.any():
                scores[no_valid, -1] = 0.0                      # no-op in the padding (NOT position 0)
            if self.mode in ("sup", "curr") and gp is not None:
                tgt = gp[:, step]
                valid = tgt >= 0
                tgt_c = torch.where(valid, torch.clamp(tgt, 0, W - 1),
                                    torch.full_like(tgt, W - 1))   # finished -> last window (padding)
                if valid.any():
                    struct_loss = struct_loss + F.cross_entropy(scores[valid], tgt_c[valid], reduction='sum')
                    struct_n += int(valid.sum())
                y_hard = torch.zeros(B, W, device=device).scatter_(1, tgt_c.unsqueeze(1), 1.0)
                soft = torch.softmax(scores, dim=1)
                y = y_hard + (soft - soft.detach())
            else:
                soft = torch.softmax(scores, dim=1)
                y_hard = torch.zeros_like(soft).scatter_(1, scores.argmax(1, keepdim=True), 1.0)
                y = y_hard + (soft - soft.detach())
            # finished examples: NO merge (y=0 -> the prefix is copied as is, root intact).
            # (bug caught by the unit test: at W=1 the "last window" IS the root)
            y = y * (~no_valid).float().unsqueeze(1)
            # differentiable reconstruction, a block of K+1 tokens is removed:
            # new[j] = a_j*old[j] + y_j*C_j + b_j*old[j+K+1], new length = cur-(K+1) = W
            csum = torch.cumsum(y, dim=1)
            a_j, b_j = 1.0 - csum, csum - y
            R = a_j.unsqueeze(2) * R[:, :W] + y.unsqueeze(2) * C + b_j.unsqueeze(2) * R[:, K + 1:]
            lengths = torch.where(lengths > 1, lengths - (K + 1), lengths)
            mask = torch.arange(R.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
        struct_loss = struct_loss / max(1, struct_n)
        return self.head(R[:, 0]), struct_loss


def unit_test_padding(device):
    """Lesson from the binary parser: check BEFORE any long run that the roots of short
    sequences survive the no-op steps when they are batched with long ones."""
    torch.manual_seed(0)
    m = SAAParserModel(d=16, hidden=32, K=5, mode="free").to(device)
    from listops_data import gen_tree
    rng = random.Random(0)
    small = serialize(gen_tree(1, 5, 0.2, rng))     # 7 tokens -> 1 reduction step
    big = serialize(gen_tree(3, 5, 0.2, rng))       # long -> several reductions
    with torch.no_grad():
        lone, _ = m([small], device)                # alone
        batched, _ = m([small, big], device)        # batched with a long one
    diff = (lone[0] - batched[0]).abs().max().item()
    assert diff < 1e-4, f"padding BUG: corrupted root (diff={diff})"
    print(f"[unit] OK - roots of short sequences intact in a batch (diff={diff:.2e})", flush=True)


@torch.no_grad()
def accuracy(model, data, device, chunk=128, teacher_force=False):
    model.eval(); correct = total = 0
    for i in range(0, len(data), chunk):
        b = data[i:i + chunk]
        gp = [g for _, _, g in b] if teacher_force else None
        logits, _ = model([s for s, _, _ in b], device, gold_positions=gp)
        pred = logits.argmax(-1)
        gold = torch.tensor([v for _, v, _ in b], device=device)
        correct += (pred == gold).sum().item(); total += len(b)
    model.train(); return correct / total


def train(model, train_data, eval_sets, device, iters, batch, lr, log_every, seed, cosine, warmup, struct_w,
          curr_hold=0.3):
    """curr_hold: 'curr' mode only. Fraction of the iterations spent in FULL teacher forcing
    before the linear weaning (lets the composer become competent before the structure is
    released; a direct test of the chicken-and-egg hypothesis: the composer must first be
    able to compute)."""
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
        b = [train_data[rng.randrange(len(train_data))] for _ in range(batch)]
        force_prob = 1.0
        if model.mode == "curr":
            hold_end = curr_hold * iters
            if it <= hold_end:
                force_prob = 1.0
            else:
                force_prob = max(0.0, 1.0 - (it - hold_end) / max(1, iters - hold_end))
        use_forcing = model.mode == "sup" or (model.mode == "curr" and rng.random() < force_prob)
        gp = [g for _, _, g in b] if use_forcing else None
        logits, struct = model([s for s, _, _ in b], device, gold_positions=gp)
        gold = torch.tensor([v for _, v, _ in b], device=device)
        loss = F.cross_entropy(logits, gold) + struct_w * struct
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched: sched.step()
        if it % log_every == 0 or it == iters:
            accs = {k: accuracy(model, ts, device) for k, ts in eval_sets.items()}
            accs["in_tf"] = accuracy(model, eval_sets["in"], device, teacher_force=True)
            fp_str = f" fp={force_prob:.2f}" if model.mode == "curr" else ""
            print(f"    it {it:5d} loss {loss.item():.3f}{fp_str} | " +
                  " ".join(f"{k} {v:.3f}" for k, v in accs.items()), flush=True)
    print(f"    [done {time.time()-t0:.0f}s]", flush=True)
    res = {k: accuracy(model, ts, device) for k, ts in eval_sets.items()}
    res["in_tf"] = accuracy(model, eval_sets["in"], device, teacher_force=True)
    return res


def make_seq_dataset(n, depths, K, p_deep, seed, max_nodes, op_ids):
    trees, _ = make_dataset(n, depths, K, p_deep, seed=seed, max_nodes=max_nodes, op_ids=op_ids)
    out = []
    for t in trees:
        s = serialize(t)
        out.append((s, t.value, gold_group_positions(s, K)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="sup,free")
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
    ap.add_argument("--struct_w", type=float, default=1.0)
    ap.add_argument("--curr_hold", type=float, default=0.3, help="curr mode: fraction of iterations in full teacher forcing before weaning")
    ap.add_argument("--cell", default="saa", choices=["saa", "vanilla", "multistat"])
    ap.add_argument("--no_pos", action="store_true", help="OOD ablation: no positional embeddings")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_saa_parser.jsonl"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unit_test_padding(device)                                  # safeguard BEFORE any long run

    from listops_data import OP_TO_ID as OPMAP
    op_ids = [OPMAP[o.strip()] for o in args.ops.split(",")] if args.ops else None
    tds = [int(x) for x in args.train_depths.split(",")]
    ods = [int(x) for x in args.ood_depths.split(",")]

    train_data = make_seq_dataset(args.n_train, tds, args.K, args.p_deep, args.data_seed, args.max_nodes, op_ids)
    eval_sets = {"in": make_seq_dataset(args.n_test, tds, args.K, args.p_deep, args.data_seed + 1, args.max_nodes, op_ids)}
    for dd in ods:
        eval_sets[f"ood{dd}"] = make_seq_dataset(args.n_test, [dd], args.K, args.p_deep, args.data_seed + 100 + dd, args.max_nodes, op_ids)

    print("=" * 64)
    print(f"SAA-Parser (K-ary groups + SAA composer) | device={device} | ops={args.ops}")
    print(f"Reference points (SM+MED): binary compositions ~0.42-0.47 | SAA on the gold tree 0.988")
    print("=" * 64, flush=True)

    done = {}
    if os.path.exists(args.out):
        for l in open(args.out):
            l = l.strip()
            if l: r = json.loads(l); done[(r["seed"], r["kind"])] = r["acc"]

    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"\n#### SEED {seed} ####", flush=True)
        for kind in [k.strip() for k in args.kinds.split(",")]:
            if (seed, kind) in done:
                print(f"[resume] {kind} seed {seed} -> skip", flush=True); continue
            torch.manual_seed(seed)
            model = SAAParserModel(d=args.d, hidden=args.hidden, K=args.K, mode=kind, cell=args.cell,
                                   use_pos=not args.no_pos)
            npar = sum(p.numel() for p in model.parameters())
            print(f"\n--- {kind} (cell={args.cell}) | params={npar:,} | seed={seed} ---", flush=True)
            res = train(model, train_data, eval_sets, device, args.iters, args.batch,
                        args.lr, args.log_every, seed, args.cosine, args.warmup, args.struct_w,
                        curr_hold=args.curr_hold)
            with open(args.out, "a") as f:
                f.write(json.dumps({"seed": seed, "kind": kind, "params": npar, "acc": res}) + "\n")
            del model
            if device == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
