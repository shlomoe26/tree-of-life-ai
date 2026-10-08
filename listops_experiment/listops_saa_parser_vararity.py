"""
Generalisation test: VARIABLE ARITY (see plans/VARIABLE_ARITY_PLAN.md).

Each node draws its arity in [Kmin, Kmax]. Serialisation uses a CANONICAL window of Kmax+2
tokens: missing operands are filled with a dedicated PAD token. Once the Kmax slots of a node
are each resolved to a single token (a leaf, a PAD, or an already folded sub-expression), the
window is always Kmax+2 tokens long, so the window scanner of listops_saa_parser.py is reused,
with an explicit PAD mask (updated through the reduction steps) that the SAA composer uses to
mask absent children.

Limits of this setup: arity is bounded by Kmax and the empty slots are visible in the input
as PAD tokens. --count feeds the number of valid children back into the composer; without it
accuracy plateaus around 0.61 (see RESULTS.md, phases 12 and 15).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import argparse, random, time, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from listops_data import make_dataset_var, NUM_DIGITS, NUM_OPS
from listops_model import SetAggCell

CLOSE = NUM_DIGITS + NUM_OPS       # 16 with six operators (14 when phase 12 was run with four)
PAD = NUM_DIGITS + NUM_OPS + 1
VOCAB = NUM_DIGITS + NUM_OPS + 2


def serialize_padded(node, Kmax):
    """Tree (variable arity, <=Kmax) -> sequence with a canonical Kmax+2 window per internal node."""
    if node.is_leaf:
        return [node.digit]
    toks = [NUM_DIGITS + node.op_id]
    for i in range(Kmax):
        if i < len(node.children):
            toks += serialize_padded(node.children[i], Kmax)
        else:
            toks.append(PAD)
    toks.append(CLOSE)
    return toks


def gold_group_positions_padded(seq, Kmax):
    """Gold order of the reductions in the padded regime: a 'resolved' slot is a value OR a PAD."""
    kinds = []
    for t in seq:
        if t == CLOSE: kinds.append('c')
        elif t == PAD: kinds.append('p')
        elif t >= NUM_DIGITS: kinds.append('o')
        else: kinds.append('v')
    pos = []
    while len(kinds) > 1:
        p = -1
        for i in range(len(kinds) - Kmax - 1):
            if kinds[i] == 'o' and all(k in ('v', 'p') for k in kinds[i + 1:i + 1 + Kmax]) and kinds[i + Kmax + 1] == 'c':
                p = i; break
        assert p >= 0, "no reducible group (malformed sequence?)"
        pos.append(p)
        kinds[p:p + Kmax + 2] = ['v']
    return pos


class VarSAAParserModel(nn.Module):
    """Like SAAParserModel, but tracks an is_pad mask through the steps so that PAD children
    are explicitly masked in the SAA composer (canonical arity Kmax, real arity variable)."""
    def __init__(self, d=64, hidden=256, Kmax=7, mode="free", max_len=400, use_pos=True, count_feat=False):
        super().__init__()
        self.d, self.Kmax, self.mode = d, Kmax, mode
        self.use_pos = use_pos
        self.tok_emb = nn.Embedding(VOCAB, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.cell = SetAggCell(d, n_heads=4, hidden=hidden, count_feat=count_feat, max_count=Kmax)
        self.query = nn.Parameter(torch.randn(d) * 0.02)
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
        is_pad = (idx == PAD).float()          # (B,Lmax): explicit tracking of the PAD status
        return R, mask, lengths, is_pad

    def forward(self, seqs, device, gold_positions=None):
        K = self.Kmax
        R, mask, lengths, is_pad = self._pack(seqs, device)
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
            W = cur - (K + 1)
            if W <= 0: break
            ops = R[:, :W]                                          # (B,W,d)
            children = R.unfold(1, K, 1)[:, 1:W + 1]                 # (B,W,d,K)
            children = children.permute(0, 1, 3, 2).reshape(B * W, K, d)
            pad_children = is_pad.unfold(1, K, 1)[:, 1:W + 1]        # (B,W,K) -> explicit mask
            pad_children = pad_children.reshape(B * W, K).bool()
            C, _ = self.cell(ops.reshape(B * W, d), children, child_mask=~pad_children)
            C = C.reshape(B, W, d)
            win_valid = mask.unfold(1, K + 2, 1)[:, :W].all(dim=2)
            scores = (C @ self.query).masked_fill(~win_valid, -1e9)
            no_valid = win_valid.sum(1) == 0
            if no_valid.any():
                scores[no_valid, -1] = 0.0
            if self.mode in ("sup", "curr") and gp is not None:
                tgt = gp[:, step]
                valid = tgt >= 0
                tgt_c = torch.where(valid, torch.clamp(tgt, 0, W - 1), torch.full_like(tgt, W - 1))
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
            y = y * (~no_valid).float().unsqueeze(1)                # finished examples -> no merge
            csum = torch.cumsum(y, dim=1)
            a_j, b_j = 1.0 - csum, csum - y
            R = a_j.unsqueeze(2) * R[:, :W] + y.unsqueeze(2) * C + b_j.unsqueeze(2) * R[:, K + 1:]
            # the composed node is never PAD (y*0); a/b carry over the PAD status of unfolded neighbours
            is_pad = a_j * is_pad[:, :W] + b_j * is_pad[:, K + 1:]
            lengths = torch.where(lengths > 1, lengths - (K + 1), lengths)
            mask = torch.arange(R.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
        struct_loss = struct_loss / max(1, struct_n)
        return self.head(R[:, 0]), struct_loss


def unit_test_padding(device):
    """Roots of short sequences stay intact when batched with long ones (a bug seen twice before)."""
    torch.manual_seed(0)
    m = VarSAAParserModel(d=16, hidden=32, Kmax=7, mode="free").to(device)
    from listops_data import gen_tree_var
    rng = random.Random(0)
    small = serialize_padded(gen_tree_var(1, 2, 7, 0.2, rng), 7)
    big = serialize_padded(gen_tree_var(3, 2, 7, 0.2, rng), 7)
    with torch.no_grad():
        lone, _ = m([small], device)
        batched, _ = m([small, big], device)
    diff = (lone[0] - batched[0]).abs().max().item()
    assert diff < 1e-4, f"batch padding BUG: corrupted root (diff={diff})"
    print(f"[unit-batch] OK - root intact in a batch (diff={diff:.2e})", flush=True)

    # Smoke check only: a group with 5 PAD children out of Kmax=7 runs through the masked
    # composer and yields finite outputs. This does NOT verify that masked children have no
    # influence on the output; that property is not unit-tested here.
    rng2 = random.Random(1)
    tsmall = gen_tree_var(1, 2, 2, 0.0, rng2)   # fixed arity 2, hence 5 PADs out of Kmax=7
    seq = serialize_padded(tsmall, 7)
    with torch.no_grad():
        out1, _ = m([seq], device)
    print(f"[unit-mask] OK - forward pass with explicit masking runs, finite output={torch.isfinite(out1).all().item()}", flush=True)


@torch.no_grad()
def accuracy(model, data, device, chunk=96, teacher_force=False):
    model.eval(); correct = total = 0
    for i in range(0, len(data), chunk):
        b = data[i:i + chunk]
        gp = [g for _, _, g in b] if teacher_force else None
        logits, _ = model([s for s, _, _ in b], device, gold_positions=gp)
        pred = logits.argmax(-1)
        gold = torch.tensor([v for _, v, _ in b], device=device)
        correct += (pred == gold).sum().item(); total += len(b)
    model.train(); return correct / total


def train(model, train_data, eval_sets, device, iters, batch, lr, log_every, seed, cosine, warmup,
          struct_w=1.0, curr_hold=0.3):
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
            force_prob = 1.0 if it <= hold_end else max(0.0, 1.0 - (it - hold_end) / max(1, iters - hold_end))
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


def make_seq_dataset(n, depths, Kmin, Kmax, p_deep, seed, max_nodes, op_ids):
    trees, _ = make_dataset_var(n, depths, Kmin, Kmax, p_deep, seed=seed, max_nodes=max_nodes, op_ids=op_ids)
    out = []
    for t in trees:
        s = serialize_padded(t, Kmax)
        out.append((s, t.value, gold_group_positions_padded(s, Kmax)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="sup,curr")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--Kmin", type=int, default=2)
    ap.add_argument("--Kmax", type=int, default=7)
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
    ap.add_argument("--curr_hold", type=float, default=0.3)
    ap.add_argument("--no_pos", action="store_true", help="no positional embeddings")
    ap.add_argument("--count", action="store_true", help="feed the number of valid operands back into the composer")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_vararity.jsonl"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unit_test_padding(device)

    from listops_data import OP_TO_ID as OPMAP
    op_ids = [OPMAP[o.strip()] for o in args.ops.split(",")] if args.ops else None
    tds = [int(x) for x in args.train_depths.split(",")]
    ods = [int(x) for x in args.ood_depths.split(",")]

    train_data = make_seq_dataset(args.n_train, tds, args.Kmin, args.Kmax, args.p_deep, args.data_seed, args.max_nodes, op_ids)
    eval_sets = {"in": make_seq_dataset(args.n_test, tds, args.Kmin, args.Kmax, args.p_deep, args.data_seed + 1, args.max_nodes, op_ids)}
    for dd in ods:
        eval_sets[f"ood{dd}"] = make_seq_dataset(args.n_test, [dd], args.Kmin, args.Kmax, args.p_deep, args.data_seed + 100 + dd, args.max_nodes, op_ids)

    lens = [len(s) for s, _, _ in train_data]
    print("=" * 64)
    print(f"VARIABLE arity [{args.Kmin},{args.Kmax}] + SAA-Parser (masked padding) | device={device}")
    print(f"train={len(train_data)} | seq length min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")
    print(f"Reference points (fixed arity K=5, with positions): sup 0.925 | curr 0.909 | free 0.660 | binary ~0.45")
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
            model = VarSAAParserModel(d=args.d, hidden=args.hidden, Kmax=args.Kmax, mode=kind,
                                      use_pos=not args.no_pos, count_feat=args.count)
            npar = sum(p.numel() for p in model.parameters())
            print(f"\n--- {kind} (Kmax={args.Kmax}) | params={npar:,} | seed={seed} ---", flush=True)
            res = train(model, train_data, eval_sets, device, args.iters, args.batch, args.lr,
                        args.log_every, seed, args.cosine, args.warmup, args.struct_w, args.curr_hold)
            with open(args.out, "a") as f:
                f.write(json.dumps({"seed": seed, "kind": kind, "params": npar, "acc": res}) + "\n")
            del model
            if device == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
