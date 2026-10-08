"""
Recursive ListOps test bench (models are GIVEN the gold syntax tree).

Modes:
  calibrate  -> safeguard 1: R0 (vanilla) ALONE. If R0 saturates in-distribution, make the
                task harder before bringing in the Etz.
  full       -> compare the cells listed in --kinds over several seeds (default: R0 vs
                Etz-1-world (E0) vs Etz-4-worlds+Tzimtzum (E4)).
                Judge no. 1 = extrapolation in depth (safeguard 2). Equal parameter budget.
  probe      -> short run of the heaviest model to read peak GPU memory before a long run.

Pre-registered verdicts for the original comparison (not to be moved afterwards):
  (Etz4 > Etz1) AND (Etz better than R0 in extrapolation) -> the structure earns its keep.
  Otherwise -> a clean null result: multiple seeds, equal budget, a discriminating judge.
Reminder: even a win would validate PRINCIPLES (hierarchy/bottleneck/sharing), NOT the
cosmology (10 Sefirot / 22 paths / 4 worlds remain unmotivated by the task).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Limit CUDA allocator fragmentation (tight VRAM on the 8 GB card used for these runs)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import argparse, random, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from listops_data import make_dataset, dataset_stats, NUM_DIGITS, OP_TO_ID, OPS
from listops_model import (VanillaCell, EtzCell, HistogramCell, MultiStatCell, GRCFoldCell, LSTMFoldCell,
                           LinearCell, ResidualCell, SetAggCell, RecursiveTreeModel,
                           count_params, find_vanilla_hidden, find_hidden)


@torch.no_grad()
def accuracy(model, trees, device, chunk=128):
    model.eval()
    correct = total = 0
    for i in range(0, len(trees), chunk):
        batch = trees[i:i + chunk]
        logits, _ = model(batch, device)
        pred = logits.argmax(-1)
        gold = torch.tensor([t.value for t in batch], device=device)
        correct += (pred == gold).sum().item()
        total += len(batch)
    model.train()
    return correct / total


@torch.no_grad()
def accuracy_by_root_op(model, trees, device, chunk=128):
    """Accuracy on `trees`, split by the operator at the root (keys 'in_SM', 'in_MED', ...)."""
    model.eval()
    hit, tot = {}, {}
    for i in range(0, len(trees), chunk):
        batch = trees[i:i + chunk]
        pred = model(batch, device)[0].argmax(-1).tolist()
        for t, p in zip(batch, pred):
            name = "in_" + OPS[t.op_id]
            tot[name] = tot.get(name, 0) + 1
            hit[name] = hit.get(name, 0) + int(p == t.value)
    model.train()
    return {k: hit[k] / tot[k] for k in sorted(tot)}


def train_model(model, train_trees, eval_sets, device, iters, batch, lr, log_every, sample_seed,
                cosine=False, warmup=200, clip=0.0):
    model.to(device).train()
    rng = random.Random(sample_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = None
    if cosine:
        import math
        def lr_lambda(it):
            if it < warmup:
                return (it + 1) / max(1, warmup)
            prog = (it - warmup) / max(1, iters - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    t0 = time.time()
    for it in range(1, iters + 1):
        bt = [train_trees[rng.randrange(len(train_trees))] for _ in range(batch)]
        logits, intent = model(bt, device)
        gold = torch.tensor([t.value for t in bt], device=device)
        loss = F.cross_entropy(logits, gold) + intent      # intent is already weighted; 0 for non-Etz cells
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if sched is not None:
            sched.step()
        if it % log_every == 0 or it == iters:
            accs = {k: accuracy(model, ts, device) for k, ts in eval_sets.items()}
            msg = " | ".join(f"{k} {v:.3f}" for k, v in accs.items())
            print(f"    it {it:5d}  loss {loss.item():.3f}  | {msg}", flush=True)
    print(f"    [done in {time.time()-t0:.1f}s]", flush=True)
    return {k: accuracy(model, ts, device) for k, ts in eval_sets.items()}


def build_datasets(args):
    train_depths = [int(x) for x in args.train_depths.split(",")]
    ood_depths = [int(x) for x in args.ood_depths.split(",")]
    op_ids = [OP_TO_ID[name.strip()] for name in args.ops.split(",")] if args.ops else None
    mn = args.max_nodes
    train_trees, _ = make_dataset(args.n_train, train_depths, args.K, args.p_deep, seed=args.data_seed, max_nodes=mn, op_ids=op_ids)
    in_trees, _ = make_dataset(args.n_test, train_depths, args.K, args.p_deep, seed=args.data_seed + 1, max_nodes=mn, op_ids=op_ids)
    eval_sets = {"in": in_trees}
    for d in ood_depths:
        ts, _ = make_dataset(args.n_test, [d], args.K, args.p_deep, seed=args.data_seed + 100 + d, max_nodes=mn, op_ids=op_ids)
        eval_sets[f"ood{d}"] = ts
    return train_trees, eval_sets, train_depths, ood_depths


def make_model(kind, args, target_params=None):
    d, K = args.d, args.K
    if kind == "R0":
        hidden = find_vanilla_hidden(target_params, d) if target_params else args.vanilla_hidden
        return RecursiveTreeModel(d, VanillaCell(d, hidden)), {"hidden": hidden}
    if kind == "Etz1":
        return RecursiveTreeModel(d, EtzCell(d, K, num_worlds=1, tzimtzum=False)), {}
    if kind == "Etz4":
        return RecursiveTreeModel(d, EtzCell(d, K, num_worlds=4, tzimtzum=True)), {}
    if kind == "Etz4_direct":   # ablation: direct weights (no hypernet)
        return RecursiveTreeModel(d, EtzCell(d, K, num_worlds=4, tzimtzum=True, weight_mode='direct')), {}
    if kind == "Etz1_direct":   # 1 world, direct weights
        return RecursiveTreeModel(d, EtzCell(d, K, num_worlds=1, tzimtzum=False, weight_mode='direct')), {}
    if kind == "Etz4_hinit":    # hypernet with calibrated initialisation (generated weights at the 'direct' scale)
        return RecursiveTreeModel(d, EtzCell(d, K, num_worlds=4, tzimtzum=True, hyper_init='calibrated')), {}
    if kind == "LSTMFold":      # binary Tree-LSTM-style fold with a memory cell, equal budget
        hidden = find_hidden(lambda h: LSTMFoldCell(d, h), target_params) if target_params else args.vanilla_hidden
        return RecursiveTreeModel(d, LSTMFoldCell(d, hidden)), {"hidden": hidden}
    if kind == "Etz4_gated":    # ablation: Tzimtzum with a learned gate
        return RecursiveTreeModel(d, EtzCell(d, K, num_worlds=4, tzimtzum=True, tzimtzum_mode='gated')), {}
    if kind == "Hist":          # compact histogram cell (predicted to win; it did not)
        return RecursiveTreeModel(d, HistogramCell(d, hidden=256)), {"hidden": 256}
    if kind == "Hist_big":      # same cell, budget matched to R0 (~208k)
        return RecursiveTreeModel(d, HistogramCell(d, hidden=400)), {"hidden": 400}
    if kind == "MultiStat":     # R0 + [mean;max;min] pooling, equal budget
        hidden = find_hidden(lambda h: MultiStatCell(d, h), target_params) if target_params else args.vanilla_hidden
        return RecursiveTreeModel(d, MultiStatCell(d, hidden)), {"hidden": hidden}
    if kind == "GRC":           # GRC-style cell (binary fold), equal budget
        hidden = find_hidden(lambda h: GRCFoldCell(d, h), target_params) if target_params else args.vanilla_hidden
        return RecursiveTreeModel(d, GRCFoldCell(d, hidden)), {"hidden": hidden}
    if kind == "Linear":        # linear composition (small by nature)
        return RecursiveTreeModel(d, LinearCell(d)), {}
    if kind == "R0_resid":      # R0 + residual, equal budget
        hidden = find_hidden(lambda h: VanillaCell(d, h), target_params) if target_params else args.vanilla_hidden
        return RecursiveTreeModel(d, ResidualCell(VanillaCell(d, hidden))), {"hidden": hidden}
    if kind == "MultiStat_resid":  # MultiStat + residual, equal budget
        hidden = find_hidden(lambda h: MultiStatCell(d, h), target_params) if target_params else args.vanilla_hidden
        return RecursiveTreeModel(d, ResidualCell(MultiStatCell(d, hidden))), {"hidden": hidden}
    if kind.startswith("SAA"):   # learned aggregator (set attention): SAA_h2 / SAA_h4 / SAA_h8 / SAA_hybrid
        hybrid = kind.endswith("hybrid")
        H = 4 if hybrid else int(kind.split("_h")[1])
        build = lambda h: SetAggCell(d, n_heads=H, hidden=h, hybrid=hybrid)
        hidden = find_hidden(build, target_params) if target_params else args.vanilla_hidden
        return RecursiveTreeModel(d, build(hidden)), {"hidden": hidden}
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calibrate", "full", "probe"], default="calibrate")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--p_deep", type=float, default=0.3)
    ap.add_argument("--train_depths", default="1,2,3,4")
    ap.add_argument("--ood_depths", default="5,6,7")
    ap.add_argument("--ops", default="", help="subset of operators, e.g. 'SM,MED' (empty = all)")
    ap.add_argument("--max_nodes", type=int, default=120)   # cap on tree size (memory + isolates depth)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--kinds", default="R0,Etz1,Etz4", help="models to run (e.g. 'Etz4_direct,Etz4_gated')")
    ap.add_argument("--budget_ref", default="Etz4", help="model whose parameter count is the budget target for the budget-matched cells")
    ap.add_argument("--cosine", action="store_true", help="cosine LR schedule + warmup")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--data_seed", type=int, default=1234)
    ap.add_argument("--vanilla_hidden", type=int, default=512)
    ap.add_argument("--per_op", action="store_true", help="also store in-distribution accuracy per root operator")
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clipping (0 = off, as in all stored runs)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_trees, eval_sets, train_depths, ood_depths = build_datasets(args)

    print("=" * 64)
    print(f"Recursive ListOps | mode={args.mode} | device={device}")
    print(f"K(fixed arity)={args.K} | d={args.d} | train_depths={train_depths} | ood={ood_depths}")
    print(f"train={len(train_trees)} stats={dataset_stats(train_trees)}")
    print("=" * 64, flush=True)

    if args.mode == "calibrate":
        # Safeguard 1: R0 alone. Is there headroom in-distribution?
        model, info = make_model("R0", args)
        print(f"R0 (vanilla) hidden={info['hidden']} | params={count_params(model):,}", flush=True)
        res = train_model(model, train_trees, eval_sets, device,
                          args.iters, args.batch, args.lr, args.log_every, sample_seed=0)
        print("-" * 64)
        print("CALIBRATION RESULT (R0 alone):")
        for k, v in res.items():
            print(f"  {k:6s} : {v:.3f}")
        in_acc = res["in"]
        if in_acc > 0.95:
            print(f"\n[!] R0 saturates in-dist ({in_acc:.3f} > 0.95). Task too easy -> make it HARDER "
                  f"(depth/arity/iters) before bringing in the Etz.")
        else:
            print(f"\n[OK] R0 leaves headroom in-dist ({in_acc:.3f}). The Etz can be brought in.")
        return

    if args.mode == "probe":
        # Memory probe: train Etz4 (the heaviest) for a few iterations + 1 eval and read the
        # peak VRAM. Goal: confirm that the run fits in free VRAM BEFORE the long run.
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model, _ = make_model("Etz4", args)
        print(f"[probe] Etz4 params={count_params(model):,} | batch={args.batch} | "
              f"max_nodes={args.max_nodes}", flush=True)
        train_model(model, train_trees, eval_sets, device,
                    args.iters, args.batch, args.lr, args.log_every, sample_seed=0)
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"\n[probe] PEAK VRAM allocated = {peak:.0f} MB (reserved "
                  f"{torch.cuda.max_memory_reserved()/1024**2:.0f} MB)")
            free, total = torch.cuda.mem_get_info()
            print(f"[probe] free VRAM now = {free/1024**2:.0f} MB / {total/1024**2:.0f} MB")
        return

    # mode full
    seeds = [int(s) for s in args.seeds.split(",")]
    # budget: budget-matched cells are sized on the parameter count of the reference model (--budget_ref)
    ref_model, _ = make_model(args.budget_ref, args)
    target = count_params(ref_model)
    print(f"Reference budget ({args.budget_ref}) = {target:,} params -> budget-matched cells are sized on it", flush=True)

    kinds = [k.strip() for k in args.kinds.split(",")]
    # results[kind][evalset] = list over seeds
    results = {k: {name: [] for name in eval_sets} for k in kinds}

    # Disk checkpoint: resume after a crash. One JSON line per finished (seed, model);
    # whatever is already done is skipped.
    done = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[(r["seed"], r["kind"])] = r["acc"]
        if done:
            print(f"[resume] {len(done)} run(s) already in checkpoint {args.out}", flush=True)

    for seed in seeds:
        print(f"\n########## SEED {seed} ##########", flush=True)
        for kind in kinds:
            if (seed, kind) in done:
                acc = done[(seed, kind)]
                print(f"[resume] {kind} seed {seed} already done -> skip ({acc})", flush=True)
                for name, v in acc.items():
                    results[kind][name].append(v)
                continue
            torch.manual_seed(seed)
            budget_kinds = {"R0", "MultiStat", "GRC", "LSTMFold", "R0_resid", "MultiStat_resid"}
            wants_budget = kind in budget_kinds or kind.startswith("SAA")
            model, info = make_model(kind, args, target_params=target if wants_budget else None)
            tag = kind + (f"(h={info.get('hidden')})" if wants_budget else "")
            print(f"\n--- {tag} | params={count_params(model):,} | seed={seed} ---", flush=True)
            res = train_model(model, train_trees, eval_sets, device,
                              args.iters, args.batch, args.lr, args.log_every, sample_seed=seed,
                              cosine=args.cosine, warmup=args.warmup, clip=args.clip)
            if args.per_op:
                per_op = accuracy_by_root_op(model, eval_sets["in"], device)
                print("    per root operator: " + " ".join(f"{k} {v:.3f}" for k, v in per_op.items()), flush=True)
            with open(args.out, "a") as f:
                row = {"seed": seed, "kind": kind, "params": count_params(model), "acc": res}
                if args.per_op:
                    row["per_op"] = per_op
                f.write(json.dumps(row) + "\n")
            for name, v in res.items():
                results[kind][name].append(v)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # aggregated summary
    def mean_std(xs):
        m = sum(xs) / len(xs)
        s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return m, s

    print("\n" + "=" * 64)
    print("SUMMARY (mean ± std over seeds)  - judge no. 1 = OOD depth")
    print("=" * 64)
    header = "model".ljust(10) + " | " + " | ".join(name.center(13) for name in eval_sets)
    print(header)
    print("-" * len(header))
    for kind in kinds:
        cells = []
        for name in eval_sets:
            m, s = mean_std(results[kind][name])
            cells.append(f"{m:.3f}±{s:.3f}".center(13))
        print(kind.ljust(10) + " | " + " | ".join(cells))
    print("=" * 64)
    print("Reminder: this tests PRINCIPLES (hierarchy/bottleneck/sharing), not the cosmology.")


if __name__ == "__main__":
    main()
