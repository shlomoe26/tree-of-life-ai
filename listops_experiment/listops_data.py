"""
ListOps generator with FIXED ARITY K and controlled DEPTH.

ListOps: nested expressions of operators over digits 0-9.
  - MAX, MIN, MED (median), SM (sum modulo 10), plus MODE and RNG (added later, see below)
  - every internal node has EXACTLY K children (fixed arity; `gen_tree_var` lifts this)
  - the output is a single digit 0-9  -> a 10-class classification task

The tree-based models (listops_run.py) are given the syntax tree (gold parse): they test
the combination CELL, not structure discovery. The parser scripts serialise the same trees
into flat sequences and hide the tree.

Pure Python (no numpy). Each node stores its height so that the torch side can evaluate
a batch of trees bucket by bucket, by height.
"""
import random

# MODE/RNG were appended at the END of the list (ids MAX0/MIN1/MED2/SM3 unchanged) for the
# "unseen operators" test: semantics of a different kind (counting, combining order statistics).
# NB: NUM_OPS went from 4 to 6, so the derived CLOSE/PAD token ids moved (+2). Stored results
# (jsonl) are unaffected, but phases 4-12 were produced with the 4-operator vocabulary: a model
# retrained today has 128 more embedding parameters and a different initialisation at equal seed.
OPS = ["MAX", "MIN", "MED", "SM", "MODE", "RNG"]   # SM = sum mod 10 ; MODE = most frequent ; RNG = max-min
OP_TO_ID = {op: i for i, op in enumerate(OPS)}
NUM_OPS = len(OPS)
NUM_DIGITS = 10


class Node:
    __slots__ = ("op_id", "digit", "children", "height", "value")

    def __init__(self, op_id=-1, digit=-1, children=None):
        self.op_id = op_id          # -1 for a leaf
        self.digit = digit          # -1 for an internal node
        self.children = children if children is not None else []
        self.height = 0             # 0 for a leaf
        self.value = None           # integer value 0-9 (gold)

    @property
    def is_leaf(self):
        return self.op_id == -1


def _apply(op_id, vals):
    op = OPS[op_id]
    if op == "MAX":
        return max(vals)
    if op == "MIN":
        return min(vals)
    if op == "MED":
        s = sorted(vals)
        return s[len(s) // 2]       # odd K -> exact middle (upper median when K is even)
    if op == "SM":
        return sum(vals) % 10
    if op == "MODE":                # most frequent value (ties -> the smallest)
        best, best_n = None, -1
        for v in sorted(set(vals)):
            c = vals.count(v)
            if c > best_n:
                best, best_n = v, c
        return best
    if op == "RNG":                 # range max-min (always within 0-9)
        return max(vals) - min(vals)
    raise ValueError(op)


def _leaf(rng):
    n = Node(digit=rng.randrange(NUM_DIGITS))
    n.height = 0
    n.value = n.digit
    return n


def gen_tree(depth, K, p_deep, rng, op_ids=None):
    """
    Generate a tree whose longest path has depth == `depth`.
    depth=0 -> leaf. EXACTLY K children on every internal node.
    At least one child reaches depth-1 (guarantees the target depth).
    op_ids: subset of allowed operators (ids). None = all.
    """
    if depth == 0:
        return _leaf(rng)

    op_id = rng.choice(op_ids) if op_ids else rng.randrange(NUM_OPS)
    deep_slot = rng.randrange(K)
    children = []
    for k in range(K):
        if k == deep_slot:
            child = gen_tree(depth - 1, K, p_deep, rng, op_ids)         # guaranteed deep branch
        elif depth - 1 > 0 and rng.random() < p_deep:
            child = gen_tree(rng.randint(1, depth - 1), K, p_deep, rng, op_ids)  # medium branch
        else:
            child = _leaf(rng)
        children.append(child)

    n = Node(op_id=op_id, children=children)
    n.height = 1 + max(c.height for c in children)
    n.value = _apply(op_id, [c.value for c in children])
    return n


def gen_tree_var(depth, Kmin, Kmax, p_deep, rng, op_ids=None):
    """
    Like gen_tree, but with VARIABLE arity: each internal node draws its own number of
    children uniformly in [Kmin, Kmax]. Generalisation test (plans/VARIABLE_ARITY_PLAN.md).
    """
    if depth == 0:
        return _leaf(rng)

    K = rng.randint(Kmin, Kmax)
    op_id = rng.choice(op_ids) if op_ids else rng.randrange(NUM_OPS)
    deep_slot = rng.randrange(K)
    children = []
    for k in range(K):
        if k == deep_slot:
            child = gen_tree_var(depth - 1, Kmin, Kmax, p_deep, rng, op_ids)
        elif depth - 1 > 0 and rng.random() < p_deep:
            child = gen_tree_var(rng.randint(1, depth - 1), Kmin, Kmax, p_deep, rng, op_ids)
        else:
            child = _leaf(rng)
        children.append(child)

    n = Node(op_id=op_id, children=children)
    n.height = 1 + max(c.height for c in children)
    n.value = _apply(op_id, [c.value for c in children])
    return n


def tree_size(n):
    if n.is_leaf:
        return 1
    return 1 + sum(tree_size(c) for c in n.children)


def make_dataset(n_samples, depths, K, p_deep, seed, max_nodes=None, op_ids=None):
    """
    Build a list of trees, drawing the depth uniformly from `depths`.
    Returns (trees, label_distribution); labels are the root value 0-9.

    max_nodes: if set, any tree with more nodes is REJECTED and regenerated.
    Two purposes: (a) it bounds GPU memory (the autograd graph grows with the node count);
    (b) it isolates DEPTH as the only variable of the extrapolation test (deep but narrow
    trees) instead of mixing depth and size.
    op_ids: subset of allowed operators (ids). None = all.
    """
    rng = random.Random(seed)
    trees = []
    label_counts = [0] * NUM_DIGITS
    while len(trees) < n_samples:
        d = rng.choice(depths)
        t = gen_tree(d, K, p_deep, rng, op_ids)
        if max_nodes is not None and tree_size(t) > max_nodes:
            continue
        trees.append(t)
        label_counts[t.value] += 1
    return trees, label_counts


def make_dataset_var(n_samples, depths, Kmin, Kmax, p_deep, seed, max_nodes=None, op_ids=None):
    """Like make_dataset, with variable arity per node (gen_tree_var). See plans/VARIABLE_ARITY_PLAN.md."""
    rng = random.Random(seed)
    trees = []
    label_counts = [0] * NUM_DIGITS
    while len(trees) < n_samples:
        d = rng.choice(depths)
        t = gen_tree_var(d, Kmin, Kmax, p_deep, rng, op_ids)
        if max_nodes is not None and tree_size(t) > max_nodes:
            continue
        trees.append(t)
        label_counts[t.value] += 1
    return trees, label_counts


def dataset_stats(trees):
    sizes = [tree_size(t) for t in trees]
    heights = [t.height for t in trees]
    return {
        "n": len(trees),
        "size_min": min(sizes), "size_max": max(sizes),
        "size_mean": sum(sizes) / len(sizes),
        "height_min": min(heights), "height_max": max(heights),
    }


if __name__ == "__main__":
    # small sanity check + readable example
    def render(n):
        if n.is_leaf:
            return str(n.digit)
        return f"{OPS[n.op_id]}(" + ", ".join(render(c) for c in n.children) + ")"

    rng = random.Random(0)
    for d in range(1, 5):
        t = gen_tree(d, K=5, p_deep=0.3, rng=rng)
        print(f"depth={d} height={t.height} size={tree_size(t)} value={t.value}")
        print("  ", render(t))
