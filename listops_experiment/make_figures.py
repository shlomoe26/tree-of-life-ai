"""
Regenerate every figure of the repository from the raw result files (results_*.jsonl).

    python listops_experiment/make_figures.py

All figures are grouped bar charts with the same two series: accuracy in distribution
(depths 1-4) and at the deepest out-of-distribution depth tested. Values are means over the
seeds present in each file (usually 3; see RESULTS.md for the exceptions). No dependency
beyond the standard library.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
C1, C2 = "#2a78d6", "#eb6834"                       # in-distribution, out-of-distribution
INK, SEC, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#8a8983", "#e7e6e2", "#fcfcfb"


def mean_acc(fname, kind, key):
    """Mean of acc[key] over the runs of `kind` in results file `fname`."""
    xs = []
    for line in open(os.path.join(HERE, fname)):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r["kind"] == kind:
                xs.append(r["acc"][key])
    assert xs, (fname, kind, key)
    return sum(xs) / len(xs)


def bar(x, y0, yv, bw, color, tip, label):
    r = 4
    d = (f"M{x},{y0} V{yv + r:.1f} Q{x},{yv:.1f} {x + r},{yv:.1f} H{x + bw - r} "
         f"Q{x + bw},{yv:.1f} {x + bw},{yv + r:.1f} V{y0} Z")
    return (f'<path d="{d}" fill="{color}"><title>{tip}: {label}</title></path>'
            f'<text x="{x + bw / 2}" y="{yv - 6:.1f}" text-anchor="middle" font-size="11" fill="{INK}">{label}</text>')


def grouped_bars(fname, title, subtitle, sections, ood_label, footer, bw=36):
    """sections = [(section name, [(label line 1, label line 2, in_acc, ood_acc), ...]), ...]"""
    n = sum(len(items) for _, items in sections)
    sect_gap, gap = 30, 2
    W, H = 940, 520
    # groups share the plot width evenly (capped so that a few groups do not spread too far)
    slot = min(170.0, (W - 60 - 40 - (len(sections) - 1) * sect_gap) / n)
    pad = (slot - 2 * bw - gap) / 2
    y0, y1 = 400, 100

    def Y(v):
        return y0 - v * (y0 - y1)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif" role="img" aria-label="{title}">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="60" y="30" font-size="18" font-weight="600" fill="{INK}">{title}</text>',
         f'<text x="60" y="52" font-size="13" fill="{SEC}">{subtitle}</text>',
         f'<rect x="60" y="66" width="12" height="12" rx="2" fill="{C1}"/><text x="78" y="77" font-size="12" fill="{SEC}">in distribution (depths 1-4)</text>',
         f'<rect x="290" y="66" width="12" height="12" rx="2" fill="{C2}"/><text x="308" y="77" font-size="12" fill="{SEC}">{ood_label}</text>']
    for v in (0, .25, .5, .75, 1.0):
        o.append(f'<line x1="60" y1="{Y(v):.1f}" x2="{W - 30}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="52" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUT}">{v:.2f}</text>')
    x = 60 + (W - 60 - 30 - n * slot - (len(sections) - 1) * sect_gap) / 2
    for sname, items in sections:
        gx0 = x + pad
        for l1, l2, a, b in items:
            tip = f"{sname} - {l1} {l2}".strip()
            bx = round(x + pad, 1)
            o.append(bar(bx, y0, Y(a), bw, C1, tip + " - in distribution", f"{a:.3f}"))
            o.append(bar(bx + bw + gap, y0, Y(b), bw, C2, tip + " - out of distribution", f"{b:.3f}"))
            cx = bx + bw + gap / 2
            o.append(f'<text x="{cx}" y="{y0 + 17}" text-anchor="middle" font-size="11.5" fill="{INK}">{l1}</text>')
            o.append(f'<text x="{cx}" y="{y0 + 31}" text-anchor="middle" font-size="11.5" fill="{INK}">{l2}</text>')
            x += slot
        gx1 = x - pad
        o.append(f'<line x1="{gx0}" y1="{y0 + 44}" x2="{gx1}" y2="{y0 + 44}" stroke="{MUT}" stroke-width="1"/>')
        o.append(f'<text x="{(gx0 + gx1) / 2}" y="{y0 + 62}" text-anchor="middle" font-size="12.5" font-weight="600" fill="{SEC}">{sname}</text>')
        x += sect_gap
    o.append(f'<line x1="60" y1="{y0}" x2="{W - 30}" y2="{y0}" stroke="{SEC}" stroke-width="1"/>')
    o.append(f'<text x="60" y="{H - 14}" font-size="11" fill="{MUT}">{footer}</text>')
    o.append('</svg>')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, fname), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(o))
    print("wrote", fname, f"({W}x{H})")


def pair(f, kind, ood="ood9"):
    return mean_acc(f, kind, "in"), mean_acc(f, kind, ood)


def main():
    # 1. Etz vs flat MLP, gold tree given, 3,000 iterations
    grouped_bars(
        "fig1_etz_vs_mlp.svg",
        "Depth helps the Etz once structure is fed in, but a flat MLP still wins",
        "Recursive cells on the gold tree, ~1.1M parameters each, 3,000 iterations, 3 seeds",
        [("All four operators (deepest OOD = depth 7)",
          [("R0", "flat MLP", *pair("results_easy.jsonl", "R0", "ood7")),
           ("Etz", "1 world", *pair("results_easy.jsonl", "Etz1", "ood7")),
           ("Etz", "4 worlds", *pair("results_easy.jsonl", "Etz4", "ood7"))]),
         ("Hard variant, SM + MED (deepest OOD = depth 9)",
          [("R0", "flat MLP", *pair("results_hard.jsonl", "R0")),
           ("Etz", "1 world", *pair("results_hard.jsonl", "Etz1")),
           ("Etz", "4 worlds", *pair("results_hard.jsonl", "Etz4"))])],
        "deepest out-of-distribution depth tested",
        "Data: results_easy.jsonl, results_hard.jsonl.")

    # 2. Autopsy
    grouped_bars(
        "fig2_autopsy.svg",
        "Removing the hypernetwork recovers the accuracy; the flat MLP is still as good",
        "Hard variant (SM + MED), gold tree, 3,000 iterations, 3 seeds. Parameter counts under each model.",
        [("Etz, 4 worlds",
          [("hypernet", "1.11M", *pair("results_hard.jsonl", "Etz4")),
           ("gated Tzimtzum", "1.11M", *pair("results_ablation.jsonl", "Etz4_gated")),
           ("direct weights", "205k", *pair("results_ablation.jsonl", "Etz4_direct"))]),
         ("Baseline",
          [("R0 flat MLP", "208k", *pair("results_followup.jsonl", "R0"))])],
        "out of distribution (depth 9)",
        "Data: results_hard.jsonl, results_ablation.jsonl, results_followup.jsonl. Etz direct weights: std 0.069 over seeds.")

    # 3. Training budget
    grouped_bars(
        "fig3_training_budget.svg",
        "The 0.79 'ceiling' was under-training: the MLP reaches 0.95, the GRC fold stays put",
        "Hard variant, gold tree, ~208k parameters. 3k = 3,000 iterations; 10k = 10,000 + cosine schedule + 50,000 examples.",
        [("Mean-pool MLP (R0)",
          [("3k", "", *pair("results_followup.jsonl", "R0")), ("10k", "", *pair("results_overnight.jsonl", "R0"))]),
         ("[mean ; max ; min] pooling",
          [("3k", "", *pair("results_multistat.jsonl", "MultiStat")), ("10k", "", *pair("results_overnight.jsonl", "MultiStat"))]),
         ("GRC-style binary fold",
          [("3k", "", *pair("results_grc3k.jsonl", "GRC")), ("10k", "", *pair("results_overnight.jsonl", "GRC"))])],
        "out of distribution (depth 9)",
        "Data: results_followup, results_multistat, results_grc3k, results_overnight (.jsonl). GRC at 10k: 2 seeds.")

    # 4. Cells at 10k
    grouped_bars(
        "fig4_cells.svg",
        "Tree given: learned aggregation beats fixed pooling; rigid cells stay near 0.4-0.5",
        "Hard variant, gold tree, 10,000 iterations + cosine + 50,000 examples, ~208k parameters unless noted.",
        [("Rigid / structured",
          [("Linear", "9.8k", *pair("results_overnight.jsonl", "Linear")),
           ("GRC fold", "", *pair("results_overnight.jsonl", "GRC")),
           ("Etz 4 worlds", "1.11M", *pair("results_consol.jsonl", "Etz4")),
           ("Histogram", "", *pair("results_consol.jsonl", "Hist_big"))]),
         ("Soft, fixed pooling",
          [("R0", "mean", *pair("results_overnight.jsonl", "R0")),
           ("MultiStat", "mean;max;min", *pair("results_overnight.jsonl", "MultiStat"))]),
         ("Soft, learned",
          [("SAA + max/min", "hybrid", *pair("results_saa.jsonl", "SAA_hybrid")),
           ("SAA", "4 heads", *pair("results_saa.jsonl", "SAA_h4"))])],
        "out of distribution (depth 9)",
        "Data: results_overnight, results_consol, results_saa (.jsonl). 2 seeds for Linear, GRC, Etz and Histogram; 3 otherwise.")

    # 5. Flat sequences: binary wall, K-ary window, curriculum
    grouped_bars(
        "fig5_flat_sequence.svg",
        "Flat sequences: binary merging plateaus near 0.45; K-ary groups + curriculum reach 0.91",
        "Hard variant (SM + MED), no tree given, ~190-215k parameters, with positional embeddings. Free parsing at test time.",
        [("Binary merges",
          [("fixed", "left-to-right", *pair("results_parser.jsonl", "l2r")),
           ("greedy", "ST", *pair("results_parser.jsonl", "soft")),
           ("Gumbel", "+ annealing", *pair("results_gumbel.jsonl", "gumbel")),
           ("supervised", "structure", *pair("results_sup_w10.jsonl", "sup"))]),
         ("K-ary, no structure signal",
          [("mean-pool", "composer", *pair("results_saap_vanilla.jsonl", "free")),
           ("SAA", "composer", *pair("results_saaparser.jsonl", "free"))]),
         ("K-ary, structure-supervised",
          [("curriculum", "", *pair("results_saap_curr.jsonl", "curr")),
           ("teacher", "forcing", *pair("results_saaparser.jsonl", "sup"))])],
        "out of distribution (depth 9)",
        "Data: results_parser, results_gumbel, results_sup_w10, results_saap_vanilla, results_saaparser, results_saap_curr (.jsonl). Left-to-right: 2 seeds.")

    # 6. Generalisation
    grouped_bars(
        "fig6_generalization.svg",
        "Without positions the depth gap closes; with the operand count, variable arity recovers",
        "SAA-Parser trained by curriculum on flat sequences, 3 seeds, 148k-192k parameters.",
        [("Fixed arity K=5, SM + MED",
          [("with", "positions", *pair("results_saap_curr.jsonl", "curr")),
           ("no", "positions", *pair("results_nopos.jsonl", "curr"))]),
         ("Variable arity 2-7, SM + MED",
          [("with", "positions", *pair("results_vararity.jsonl", "curr")),
           ("no", "positions", *pair("results_vararity_nopos.jsonl", "curr")),
           ("no positions", "+ count", *pair("results_var_count.jsonl", "curr"))]),
         ("Other operators, K=5",
          [("MODE + RNG", "with positions", *pair("results_newops.jsonl", "curr")),
           ("6 operators", "no positions", *pair("results_capstone.jsonl", "curr"))])],
        "out of distribution (depth 9)",
        "Data: results_saap_curr, results_nopos, results_vararity, results_vararity_nopos, results_var_count, results_newops, results_capstone (.jsonl).")


if __name__ == "__main__":
    main()
