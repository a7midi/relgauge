"""
sharedsquareholonomy.py -- fully shared-corner square holonomy test.

This is the stricter successor to squareholonomy.py.  In squareholonomy.py, the
four edges of a square were represented by four separable temporal-chain edge
modules generated together.  Here the four edges live in ONE microscopic finite
system with shared corner variables:

        A --AB--> B --BD--> D
        |                    ^
        AC                   CD
        v                    |
        C -------------------+

The same B corner state is both the sink of AB and the source of BD; likewise C.
The same A source feeds both outgoing edges, and the same D corner receives both
incoming edges.  Each directed edge is implemented as a one-diamond transporter:
source boundary -> two branch vertices -> two sink panels.  Intermediate corner
outputs B_out and C_out are computed from the incoming corner panels and then
feed the second-step edges.

For a square instance we extract exact equalizer labels at the four edge sinks:
    S_AB at B, S_AC at C, S_BD at D(top), S_CD at D(bottom).
We define a common A-label by S_AB and infer maps
    phi_AB : S_A  -> S_B       (identity by construction)
    phi_AC : S_A  -> S_C
    phi_BD : S_B  -> S_D_top
    phi_CD : S_C  -> S_D_bottom.
When these maps are deterministic bijections on a common label alphabet, the
path curvature is
    Delta = (phi_BD o phi_AB)^(-1) o (phi_CD o phi_AC).

Under local relabelings g_A,g_B,g_C,g_D, Delta transforms as
    Delta -> g_A Delta g_A^{-1},
so its conjugacy class is the finite gauge-invariant observable.

Recommended run
---------------
python -m relgauge.sharedsquareholonomy 4 ^
  --ws 1 ^
  --instances 50 ^
  --ensembles copy,permutive,block,canalizing,random ^
  --mutation-rates 0,0.02,0.05 ^
  --initial-mode joint_random ^
  --n-random-initial 4096 ^
  --out example_results/shared_square_holonomy_q4_w1.csv ^
  --plot example_results/fig_shared_square_holonomy_q4_w1.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

from . import core as C

Ensemble = Literal["random", "copy", "permutive", "block", "canalizing", "constant"]
InitialMode = Literal["source_all", "source_random", "joint_random"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _all_inputs(indeg: int, q: int) -> np.ndarray:
    return C.all_states(int(indeg), int(q)) if indeg else np.zeros((1, 0), dtype=np.int64)


def _source_pos(preds: tuple[int, ...], src: int) -> int:
    try:
        return tuple(int(x) for x in preds).index(int(src))
    except ValueError as exc:
        raise ValueError(f"source {src} not in preds {preds}") from exc


def _rule_from_source(q: int, indeg: int, source_pos: int, transform: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    X = _all_inputs(indeg, q)
    y = X[:, int(source_pos)]
    if transform is not None:
        y = transform(y)
    return np.asarray(y, dtype=np.int64) % int(q)


def _constant_rule(q: int, indeg: int, value: int = 0) -> np.ndarray:
    return np.full(q ** int(indeg), int(value) % int(q), dtype=np.int64)


def _random_rule(q: int, indeg: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, q, size=q ** int(indeg), dtype=np.int64)


def _set_copy(sys: C.RelationalSystem, target: int, source: int, transform: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
    preds = sys.preds[int(target)]
    pos = _source_pos(preds, int(source))
    sys.rules[int(target)] = _rule_from_source(sys.q, len(preds), pos, transform)


def _set_constant(sys: C.RelationalSystem, target: int, value: int = 0) -> None:
    sys.rules[int(target)] = _constant_rule(sys.q, len(sys.preds[int(target)]), value)


def _mutate_rule(rule: np.ndarray, q: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(rule, dtype=np.int64).copy()
    if rate <= 0:
        return out
    mask = rng.random(out.size) < float(rate)
    if np.any(mask):
        jumps = rng.integers(1, q, size=int(mask.sum()), dtype=np.int64)
        out[mask] = (out[mask] + jumps) % int(q)
    return out


def _perm_transform(perm: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    p = np.asarray(perm, dtype=np.int64)
    return lambda x: p[np.asarray(x, dtype=np.int64)]


def _block_transform(q: int, n_blocks: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    if n_blocks is None:
        n_blocks = max(2, min(int(q), int(round(math.sqrt(q)))))
    n_blocks = max(1, min(int(n_blocks), int(q)))
    return lambda x: (np.asarray(x, dtype=np.int64) * n_blocks // int(q)).astype(np.int64)


def _canalizing_transform(q: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: (np.asarray(x, dtype=np.int64) != 0).astype(np.int64)


def _encode_words_from_state(S: np.ndarray, vertices: Iterable[int], q: int) -> np.ndarray:
    verts = tuple(int(v) for v in vertices)
    if not verts:
        return np.zeros(S.shape[0], dtype=np.int64)
    out = np.zeros(S.shape[0], dtype=np.int64)
    for j, v in enumerate(verts):
        out += S[:, v].astype(np.int64) * (int(q) ** j)
    return out.astype(np.int64)


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _entropy_discrete(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    _, counts = np.unique(np.asarray(x, dtype=np.int64), return_counts=True)
    return _entropy_from_counts(counts)


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if x.shape != y.shape or len(x) == 0:
        return 0.0
    xv, xi = np.unique(x, return_inverse=True)
    yv, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xv), len(yv)), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)
    return float(_entropy_from_counts(joint.sum(axis=1)) + _entropy_from_counts(joint.sum(axis=0)) - _entropy_from_counts(joint.reshape(-1)))



class _UF:
    def __init__(self, n: int):
        self.p = list(range(int(n)))
    def find(self, x: int) -> int:
        x = int(x); r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def exact_equalizer_labels(left_words: np.ndarray, right_words: np.ndarray, q: int, w: int) -> dict:
    left_words = np.asarray(left_words, dtype=np.int64)
    right_words = np.asarray(right_words, dtype=np.int64)
    if left_words.shape != right_words.shape:
        raise ValueError("left and right must have same shape")
    n_side = int(q ** w)
    if len(left_words) == 0:
        return dict(shared_classes=0, residual_qcoords=math.nan, labels=np.array([], dtype=np.int64), edge_density=math.nan)
    uf = _UF(2 * n_side)
    pairs = np.unique(np.stack([left_words, right_words], axis=1), axis=0)
    for l, r in pairs:
        uf.union(int(l), n_side + int(r))
    roots = sorted({uf.find(int(l)) for l in left_words} | {uf.find(n_side + int(r)) for r in right_words})
    root_to_label = {root: i for i, root in enumerate(roots)}
    labels = np.array([root_to_label[uf.find(int(l))] for l in left_words], dtype=np.int64)
    shared = len(roots)
    return dict(
        shared_classes=int(shared),
        residual_qcoords=float(math.log(shared, q)) if shared > 0 else math.nan,
        labels=labels,
        edge_density=float(len(pairs) / (n_side * n_side)),
        pairs=np.asarray(pairs, dtype=np.int64),
    )


def compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(p[int(q[i])]) for i in range(len(p)))


def identity(n: int) -> tuple[int, ...]:
    return tuple(range(int(n)))


def inverse(p: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[int(j)] = int(i)
    return tuple(inv)


def cycle_lengths(p: tuple[int, ...]) -> tuple[int, ...]:
    n = len(p); seen = [False] * n; out = []
    for i in range(n):
        if seen[i]:
            continue
        cur = i; l = 0
        while not seen[cur]:
            seen[cur] = True; l += 1; cur = int(p[cur])
        out.append(l)
    return tuple(sorted(out, reverse=True))


def cycle_type_name(p: tuple[int, ...] | None) -> str:
    if p is None:
        return "nonpermutation"
    ct = cycle_lengths(tuple(p))
    if all(x == 1 for x in ct):
        return "identity"
    return "cycle_" + "_".join(str(x) for x in ct)


def conjugate(g: tuple[int, ...], h: tuple[int, ...]) -> tuple[int, ...]:
    return compose(compose(g, h), inverse(g))


def _perm_order(p: tuple[int, ...]) -> int:
    lcm = 1
    for l in cycle_lengths(p):
        lcm = abs(lcm * int(l)) // math.gcd(lcm, int(l))
    return int(lcm)


def _parity(p: tuple[int, ...]) -> int:
    inv = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            inv += int(p[i] > p[j])
    return inv % 2


def generate_group(gens: Iterable[tuple[int, ...]], n: int, max_size: int = 100000) -> set[tuple[int, ...]]:
    from collections import deque
    gens = [tuple(int(x) for x in g) for g in gens if len(g) == int(n)]
    ident = identity(n)
    if not gens:
        return {ident}
    group = {ident}; queue = deque([ident]); gen_set = list({*gens, ident})
    while queue:
        a = queue.popleft()
        for g in gen_set:
            for h in (compose(a, g), compose(g, a)):
                if h not in group:
                    group.add(h); queue.append(h)
                    if len(group) > int(max_size):
                        return group
    return group


def classify_group(group: set[tuple[int, ...]], n: int) -> dict:
    order = len(group)
    orders = Counter(_perm_order(g) for g in group)
    cycle_types = Counter(str(cycle_lengths(g)) for g in group)
    all_even = all(_parity(g) == 0 for g in group)
    glist = list(group); abelian = True
    for i, a in enumerate(glist):
        for b in glist[i+1:]:
            if compose(a,b) != compose(b,a):
                abelian = False; break
        if not abelian: break
    cyclic = any(_perm_order(g) == order for g in group)
    orbit = {int(g[0]) for g in group} if group else set()
    transitive = len(orbit) == int(n)
    name = f"order_{order}"
    if order == 1: name = "trivial"
    elif order == 2 and cyclic: name = "C2"
    elif order == 3 and cyclic: name = "C3"
    elif order == 4: name = "C4" if cyclic else "V4"
    elif n == 4 and order == 8: name = "D4_or_order8"
    elif n == 4 and order == 12 and all_even: name = "A4"
    elif n == 4 and order == 24: name = "S4"
    elif cyclic: name = f"C{order}"
    return dict(group_name=name, group_order=int(order), group_cyclic=bool(cyclic), group_abelian=bool(abelian), group_transitive=bool(transitive), group_all_even=bool(all_even), element_order_hist={str(k): int(v) for k,v in sorted(orders.items())}, cycle_type_hist={str(k): int(v) for k,v in sorted(cycle_types.items())})


# --------------------------------------------------------------------------- #
# Shared-corner construction
# --------------------------------------------------------------------------- #
@dataclass
class SharedSquare:
    joint: C.RelationalSystem
    q: int
    w: int
    A_src: tuple[int, ...]
    B_left: tuple[int, ...]
    B_right: tuple[int, ...]
    B_out: tuple[int, ...]
    C_left: tuple[int, ...]
    C_right: tuple[int, ...]
    C_out: tuple[int, ...]
    D_top_left: tuple[int, ...]
    D_top_right: tuple[int, ...]
    D_bottom_left: tuple[int, ...]
    D_bottom_right: tuple[int, ...]
    schedule: tuple[int, ...]
    meta: dict

    @property
    def k_total(self) -> int:
        return self.joint.k


def _add_vertices(n: int, pred_sets: list[set[int]]) -> tuple[int, ...]:
    off = len(pred_sets)
    for _ in range(int(n)):
        pred_sets.append(set())
    return tuple(range(off, off + int(n)))


def make_shared_square(
    q: int,
    w: int,
    rng: np.random.Generator,
    ensemble: Ensemble = "permutive",
    mutation_rate: float = 0.0,
    n_blocks: int | None = None,
    mutate_all_rules: bool = True,
) -> SharedSquare:
    """Build one monolithic shared-corner square.

    Four edge diamonds share corner variables.  The graph is feed-forward:
    A -> {B,C} -> D.  The B and C corner outputs are computed from their own
    incoming panels, so B and C are genuinely shared between incoming and
    outgoing edge transports.
    """
    q = int(q); w = int(w)
    if w <= 0:
        raise ValueError("w must be positive")
    pred_sets: list[set[int]] = []

    # Corners.
    A_src = _add_vertices(w, pred_sets)
    B_left = _add_vertices(w, pred_sets)
    B_right = _add_vertices(w, pred_sets)
    B_out = _add_vertices(w, pred_sets)
    C_left = _add_vertices(w, pred_sets)
    C_right = _add_vertices(w, pred_sets)
    C_out = _add_vertices(w, pred_sets)
    D_top_left = _add_vertices(w, pred_sets)
    D_top_right = _add_vertices(w, pred_sets)
    D_bottom_left = _add_vertices(w, pred_sets)
    D_bottom_right = _add_vertices(w, pred_sets)

    # Branch vertices for each directed edge and each side of the edge diamond.
    branches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for name in ("AB", "AC", "BD", "CD"):
        branches[name] = (_add_vertices(w, pred_sets), _add_vertices(w, pred_sets))

    def connect(srcs: tuple[int, ...], tgts: tuple[int, ...]) -> None:
        for s, t in zip(srcs, tgts):
            pred_sets[int(t)].add(int(s))

    # Edge AB: A -> branch -> B panels.
    connect(A_src, branches["AB"][0]); connect(A_src, branches["AB"][1])
    connect(branches["AB"][0], B_left); connect(branches["AB"][1], B_right)
    # Shared B corner output from B's incoming left panel.
    connect(B_left, B_out)

    # Edge AC.
    connect(A_src, branches["AC"][0]); connect(A_src, branches["AC"][1])
    connect(branches["AC"][0], C_left); connect(branches["AC"][1], C_right)
    connect(C_left, C_out)

    # Edge BD: B_out -> branch -> D top panels.
    connect(B_out, branches["BD"][0]); connect(B_out, branches["BD"][1])
    connect(branches["BD"][0], D_top_left); connect(branches["BD"][1], D_top_right)

    # Edge CD: C_out -> branch -> D bottom panels.
    connect(C_out, branches["CD"][0]); connect(C_out, branches["CD"][1])
    connect(branches["CD"][0], D_bottom_left); connect(branches["CD"][1], D_bottom_right)

    preds = [tuple(sorted(s)) for s in pred_sets]
    rules = [_random_rule(q, len(preds[v]), rng) for v in range(len(preds))]
    sys = C.RelationalSystem(len(preds), q, preds, rules, meta=dict(ensemble="shared_square", w=w))

    # Source vertices have no predecessors and therefore are identity under the
    # core sequential update.  Set all other rules by ensemble.
    def set_targets(tgts: tuple[int, ...], srcs: tuple[int, ...], transform: Callable[[np.ndarray], np.ndarray] | None) -> None:
        for s, t in zip(srcs, tgts):
            _set_copy(sys, int(t), int(s), transform)

    def set_const(tgts: tuple[int, ...], value: int = 0) -> None:
        for t in tgts:
            _set_constant(sys, int(t), int(value))

    transforms: dict[str, object] = {"mode": ensemble}
    if ensemble == "copy":
        # All edge/corner maps identity: flat transport expected.
        edge_transforms = {name: (None, None) for name in branches}
        corner_B = corner_C = None
    elif ensemble == "permutive":
        edge_transforms = {}
        meta_perms = {}
        for name in branches:
            pL = rng.permutation(q).astype(np.int64)
            pR = rng.permutation(q).astype(np.int64)
            edge_transforms[name] = (_perm_transform(pL), _perm_transform(pR))
            meta_perms[f"{name}_L"] = pL.tolist(); meta_perms[f"{name}_R"] = pR.tolist()
        pB = rng.permutation(q).astype(np.int64)
        pC = rng.permutation(q).astype(np.int64)
        corner_B = _perm_transform(pB); corner_C = _perm_transform(pC)
        meta_perms["B_corner"] = pB.tolist(); meta_perms["C_corner"] = pC.tolist()
        transforms.update(meta_perms)
    elif ensemble == "block":
        # Coarsen once on outgoing A edges, then copy through B/C and to D.
        blk = _block_transform(q, n_blocks)
        edge_transforms = {"AB": (blk, blk), "AC": (blk, blk), "BD": (None, None), "CD": (None, None)}
        corner_B = corner_C = None
    elif ensemble == "canalizing":
        can = _canalizing_transform(q)
        edge_transforms = {"AB": (can, can), "AC": (can, can), "BD": (None, None), "CD": (None, None)}
        corner_B = corner_C = None
    elif ensemble == "constant":
        edge_transforms = {name: (None, None) for name in branches}
        corner_B = corner_C = None
    elif ensemble == "random":
        edge_transforms = {name: (None, None) for name in branches}
        corner_B = corner_C = None
    else:
        raise ValueError(f"unknown ensemble {ensemble!r}")

    if ensemble == "constant":
        for name in branches:
            set_const(branches[name][0]); set_const(branches[name][1])
        set_const(B_left); set_const(B_right); set_const(B_out)
        set_const(C_left); set_const(C_right); set_const(C_out)
        set_const(D_top_left); set_const(D_top_right); set_const(D_bottom_left); set_const(D_bottom_right)
    elif ensemble != "random":
        # Branch rules.
        for name, (lefts, rights) in branches.items():
            srcs = A_src if name in ("AB", "AC") else (B_out if name == "BD" else C_out)
            tL, tR = edge_transforms[name]
            set_targets(lefts, srcs, tL)
            set_targets(rights, srcs, tR)
        # Panels copy branch values.
        set_targets(B_left, branches["AB"][0], None); set_targets(B_right, branches["AB"][1], None)
        set_targets(C_left, branches["AC"][0], None); set_targets(C_right, branches["AC"][1], None)
        set_targets(D_top_left, branches["BD"][0], None); set_targets(D_top_right, branches["BD"][1], None)
        set_targets(D_bottom_left, branches["CD"][0], None); set_targets(D_bottom_right, branches["CD"][1], None)
        # Shared corners: outgoing B/C state is computed from incoming panel.
        set_targets(B_out, B_left, corner_B)
        set_targets(C_out, C_left, corner_C)

    if mutation_rate > 0:
        # Mutate all non-source rules or only interface rules.  In this module
        # every non-source rule is part of the shared plaquette interface.
        for v in range(sys.k):
            if len(sys.preds[v]) > 0:
                sys.rules[v] = _mutate_rule(sys.rules[v], q, mutation_rate, rng)

    # Topological schedule.
    sched: list[int] = []
    sched.extend(A_src)
    # First layer branches.
    sched.extend(branches["AB"][0]); sched.extend(branches["AB"][1])
    sched.extend(branches["AC"][0]); sched.extend(branches["AC"][1])
    # B/C panels.
    sched.extend(B_left); sched.extend(B_right); sched.extend(C_left); sched.extend(C_right)
    # B/C shared outputs.
    sched.extend(B_out); sched.extend(C_out)
    # Second layer branches.
    sched.extend(branches["BD"][0]); sched.extend(branches["BD"][1])
    sched.extend(branches["CD"][0]); sched.extend(branches["CD"][1])
    # D panels.
    sched.extend(D_top_left); sched.extend(D_top_right); sched.extend(D_bottom_left); sched.extend(D_bottom_right)
    # Include any vertices not listed just in case.
    seen = set(sched)
    sched.extend(v for v in range(sys.k) if v not in seen)

    meta = dict(sys.meta)
    meta.update(rule_ensemble=ensemble, mutation_rate=float(mutation_rate), transforms=transforms)
    return SharedSquare(
        joint=sys,
        q=q,
        w=w,
        A_src=tuple(A_src),
        B_left=tuple(B_left), B_right=tuple(B_right), B_out=tuple(B_out),
        C_left=tuple(C_left), C_right=tuple(C_right), C_out=tuple(C_out),
        D_top_left=tuple(D_top_left), D_top_right=tuple(D_top_right),
        D_bottom_left=tuple(D_bottom_left), D_bottom_right=tuple(D_bottom_right),
        schedule=tuple(int(x) for x in sched),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Simulation and map extraction
# --------------------------------------------------------------------------- #
def _initial_states(sq: SharedSquare, mode: InitialMode, n_random_initial: int, rng: np.random.Generator) -> np.ndarray:
    q = sq.q; w = sq.w; k = sq.joint.k
    source_words = np.arange(q ** w, dtype=np.int64)
    source_digits = np.zeros((len(source_words), w), dtype=np.int64)
    for j in range(w):
        source_digits[:, j] = (source_words // (q ** j)) % q
    if mode == "source_all":
        S = np.zeros((len(source_words), k), dtype=np.int64)
        S[:, list(sq.A_src)] = source_digits
        return S
    if mode == "source_random":
        N = max(int(n_random_initial), len(source_words))
        S = np.zeros((N, k), dtype=np.int64)
        reps = np.resize(source_words, N)
        for j, v in enumerate(sq.A_src):
            S[:, int(v)] = (reps // (q ** j)) % q
        return S
    if mode == "joint_random":
        N = max(int(n_random_initial), len(source_words))
        S = rng.integers(0, q, size=(N, k), dtype=np.int64)
        reps = np.resize(source_words, N)
        # Ensure every source word is represented evenly enough.
        rng.shuffle(reps)
        for j, v in enumerate(sq.A_src):
            S[:, int(v)] = (reps // (q ** j)) % q
        return S
    raise ValueError(f"unknown initial mode {mode!r}")


def _step_states(sys: C.RelationalSystem, S: np.ndarray, schedule: tuple[int, ...]) -> np.ndarray:
    S = np.asarray(S, dtype=np.int64).copy()
    for v in schedule:
        preds = sys.preds[int(v)]
        if not preds:
            continue
        vals = S[:, list(preds)]
        powers = sys.q ** np.arange(len(preds), dtype=np.int64)
        idx = (vals * powers).sum(axis=1).astype(np.int64)
        S[:, int(v)] = sys.rules[int(v)][idx]
    return S


def _edge_labels(left: np.ndarray, right: np.ndarray, q: int, w: int) -> dict:
    return exact_equalizer_labels(left, right, q, w)


def _best_permutation_map(x: np.ndarray, y: np.ndarray) -> dict:
    """Infer y = p[x].  Returns a permutation iff deterministic and bijective."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if x.shape != y.shape or len(x) == 0:
        return dict(accuracy=0.0, deterministic=False, permutation=None, n=0, mi_over_Hy=0.0)
    hx = _entropy_discrete(x); hy = _entropy_discrete(y); mi = _mutual_info_discrete(x, y)
    acc_count = 0
    mapping: dict[int, int] = {}
    deterministic = True
    for a in sorted(np.unique(x).tolist()):
        ys = y[x == a]
        vals, counts = np.unique(ys, return_counts=True)
        j = int(np.argmax(counts))
        mapping[int(a)] = int(vals[j])
        acc_count += int(counts[j])
        if len(vals) > 1:
            deterministic = False
    acc = float(acc_count / len(x))
    xs = sorted(np.unique(x).tolist())
    ys_unique = sorted(np.unique(y).tolist())
    # Reindex labels to 0..n-1 in their sorted observed order.
    x_to_i = {int(a): i for i, a in enumerate(xs)}
    y_to_i = {int(a): i for i, a in enumerate(ys_unique)}
    p = None
    if deterministic and len(xs) == len(ys_unique) and set(mapping.keys()) == set(xs) and set(mapping.values()) == set(ys_unique):
        pp = [0] * len(xs)
        for a, b in mapping.items():
            pp[x_to_i[int(a)]] = y_to_i[int(b)]
        if sorted(pp) == list(range(len(pp))):
            p = tuple(int(z) for z in pp)
    return dict(
        accuracy=acc,
        deterministic=bool(deterministic),
        permutation=p,
        n=int(len(xs)) if p is not None else int(max(len(xs), len(ys_unique))),
        Hx=float(hx), Hy=float(hy), mi=float(mi), mi_over_Hy=float(mi / hy) if hy > 1e-12 else 0.0,
    )


def _path_delta(ab: tuple[int, ...], bd: tuple[int, ...], ac: tuple[int, ...], cd: tuple[int, ...]) -> dict:
    top = compose(bd, ab)
    bottom = compose(cd, ac)
    delta = compose(inverse(top), bottom)
    return dict(top=top, bottom=bottom, delta=delta)


def _gauge_covariance(ab: tuple[int, ...], bd: tuple[int, ...], ac: tuple[int, ...], cd: tuple[int, ...], rng: np.random.Generator) -> dict:
    n = len(ab)
    # Use full symmetric random local relabelings, not just the generated subgroup.
    def rp() -> tuple[int, ...]:
        return tuple(int(x) for x in rng.permutation(n).tolist())
    gA, gB, gC, gD = rp(), rp(), rp(), rp()
    raw = _path_delta(ab, bd, ac, cd)
    abp = compose(gB, compose(ab, inverse(gA)))
    bdp = compose(gD, compose(bd, inverse(gB)))
    acp = compose(gC, compose(ac, inverse(gA)))
    cdp = compose(gD, compose(cd, inverse(gC)))
    trans = _path_delta(abp, bdp, acp, cdp)
    expected = conjugate(gA, raw["delta"])
    return dict(
        covariant=bool(trans["delta"] == expected),
        same_conjugacy_type=bool(cycle_lengths(trans["delta"]) == cycle_lengths(raw["delta"])),
    )


def measure_shared_square(
    sq: SharedSquare,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    min_source: float = 0.80,
    min_transport: float = 0.80,
    min_accuracy: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    S0 = _initial_states(sq, initial_mode, n_random_initial, rng)
    S1 = _step_states(sq.joint, S0, sq.schedule)

    source_raw = _encode_words_from_state(S0, sq.A_src, sq.q)
    B_l = _encode_words_from_state(S1, sq.B_left, sq.q); B_r = _encode_words_from_state(S1, sq.B_right, sq.q)
    C_l = _encode_words_from_state(S1, sq.C_left, sq.q); C_r = _encode_words_from_state(S1, sq.C_right, sq.q)
    DT_l = _encode_words_from_state(S1, sq.D_top_left, sq.q); DT_r = _encode_words_from_state(S1, sq.D_top_right, sq.q)
    DB_l = _encode_words_from_state(S1, sq.D_bottom_left, sq.q); DB_r = _encode_words_from_state(S1, sq.D_bottom_right, sq.q)

    eqB = _edge_labels(B_l, B_r, sq.q, sq.w)
    eqC = _edge_labels(C_l, C_r, sq.q, sq.w)
    eqDT = _edge_labels(DT_l, DT_r, sq.q, sq.w)
    eqDB = _edge_labels(DB_l, DB_r, sq.q, sq.w)
    SB = np.asarray(eqB["labels"], dtype=np.int64)
    SC = np.asarray(eqC["labels"], dtype=np.int64)
    SDT = np.asarray(eqDT["labels"], dtype=np.int64)
    SDB = np.asarray(eqDB["labels"], dtype=np.int64)

    # A's path-source label is the factor selected by AB.  This allows coarse
    # sectors (block/canalizing) to be treated as genuine lower-cardinality labels.
    SA = SB.copy()
    phi_AB = tuple(range(len(np.unique(SA)))) if len(np.unique(SA)) == len(np.unique(SB)) else None
    mAC = _best_permutation_map(SA, SC)
    mBD = _best_permutation_map(SB, SDT)
    mCD = _best_permutation_map(SC, SDB)
    source_live = _mutual_info_discrete(source_raw, SA) / max(_entropy_discrete(SA), 1e-12)
    n_classes = int(len(np.unique(SA)))
    maps = [phi_AB, mBD["permutation"], mAC["permutation"], mCD["permutation"]]
    all_bijective = bool(n_classes > 1 and all(p is not None and len(p) == n_classes for p in maps))
    min_edge_transport_val = float(min(mAC["mi_over_Hy"], mBD["mi_over_Hy"], mCD["mi_over_Hy"]))
    min_acc = float(min(mAC["accuracy"], mBD["accuracy"], mCD["accuracy"]))
    valid = bool(all_bijective and source_live >= float(min_source) and min_edge_transport_val >= float(min_transport) and min_acc >= float(min_accuracy))

    row = dict(
        q=int(sq.q), w=int(sq.w), k_total=int(sq.k_total),
        rule_ensemble=str(sq.meta.get("rule_ensemble", "unknown")),
        mutation_rate=float(sq.meta.get("mutation_rate", math.nan)),
        exact_B_classes=int(eqB["shared_classes"]),
        exact_C_classes=int(eqC["shared_classes"]),
        exact_Dtop_classes=int(eqDT["shared_classes"]),
        exact_Dbottom_classes=int(eqDB["shared_classes"]),
        source_label_classes=int(n_classes),
        source_residual_qcoords=float(math.log(n_classes, sq.q)) if n_classes > 0 else math.nan,
        source_liveness=float(source_live),
        phi_AC_transport=float(mAC["mi_over_Hy"]), phi_BD_transport=float(mBD["mi_over_Hy"]), phi_CD_transport=float(mCD["mi_over_Hy"]),
        min_edge_transport=float(min_edge_transport_val),
        min_edge_accuracy=float(min_acc),
        valid_square=bool(valid),
    )
    if not valid:
        row.update(
            delta_type="invalid",
            nontrivial_path_holonomy=False,
            top_type="invalid",
            bottom_type="invalid",
            generated_group_name="invalid",
            generated_group_order=0,
            gauge_covariance_success=math.nan,
            conjugacy_class_success=math.nan,
        )
        return row

    ab = tuple(int(x) for x in phi_AB)  # type: ignore[arg-type]
    bd = tuple(int(x) for x in mBD["permutation"])
    ac = tuple(int(x) for x in mAC["permutation"])
    cd = tuple(int(x) for x in mCD["permutation"])
    pd = _path_delta(ab, bd, ac, cd)
    delta = pd["delta"]
    group = generate_group([ab, bd, ac, cd], n=n_classes)
    gsum = classify_group(group, n=n_classes)
    cov = _gauge_covariance(ab, bd, ac, cd, rng)
    row.update(
        phi_AB=json.dumps(list(ab)), phi_BD=json.dumps(list(bd)), phi_AC=json.dumps(list(ac)), phi_CD=json.dumps(list(cd)),
        top=json.dumps(list(pd["top"])), bottom=json.dumps(list(pd["bottom"])), delta=json.dumps(list(delta)),
        top_type=cycle_type_name(pd["top"]), bottom_type=cycle_type_name(pd["bottom"]), delta_type=cycle_type_name(delta),
        nontrivial_path_holonomy=bool(delta != identity(n_classes)),
        generated_group_name=str(gsum.get("group_name")),
        generated_group_order=int(gsum.get("group_order", 0)),
        generated_group_abelian=bool(gsum.get("group_abelian", False)),
        generated_group_cyclic=bool(gsum.get("group_cyclic", False)),
        generated_group_transitive=bool(gsum.get("group_transitive", False)),
        gauge_covariance_success=bool(cov["covariant"]),
        conjugacy_class_success=bool(cov["same_conjugacy_type"]),
    )
    return row


# --------------------------------------------------------------------------- #
# Sweeps / analysis
# --------------------------------------------------------------------------- #
def run_shared_square_sweep(
    q: int = 4,
    ws: Iterable[int] = (1,),
    ensembles: Iterable[Ensemble] = ("copy", "permutive", "block", "canalizing", "random"),
    mutation_rates: Iterable[float] = (0.0, 0.02, 0.05),
    instances: int = 50,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    base_seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for w in ws:
        for ens in ensembles:
            for mu in mutation_rates:
                for inst in range(int(instances)):
                    seed = (int(base_seed) * 1000003 + int(q) * 10007 + int(w) * 1009 + hash(str(ens)) % 1000 * 37 + int(round(float(mu) * 10000)) * 11 + inst) % 2**32
                    rng = np.random.default_rng(seed)
                    sq = make_shared_square(q=int(q), w=int(w), rng=rng, ensemble=ens, mutation_rate=float(mu))
                    row = measure_shared_square(sq, initial_mode=initial_mode, n_random_initial=int(n_random_initial), rng=rng)
                    row.update(seed=int(seed), instance=int(inst), initial_mode=initial_mode, n_initial_states=int(max(n_random_initial, q ** w)))
                    rows.append(row)
                if verbose:
                    print(f"shared-square ensemble={ens}, mu={mu}, w={w} done", flush=True)
    return pd.DataFrame(rows)


def analyze_shared_square(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    valid = df[df["valid_square"].astype(bool)] if "valid_square" in df else df.iloc[0:0]
    nontriv_frac = float(valid["nontrivial_path_holonomy"].mean()) if len(valid) else 0.0
    valid_frac = float(df["valid_square"].mean()) if "valid_square" in df else 0.0
    cov = float(valid["gauge_covariance_success"].dropna().mean()) if len(valid) and "gauge_covariance_success" in valid else math.nan
    conj = float(valid["conjugacy_class_success"].dropna().mean()) if len(valid) and "conjugacy_class_success" in valid else math.nan
    by = []
    for (ens, mu, w), g in df.groupby(["rule_ensemble", "mutation_rate", "w"]):
        gv = g[g["valid_square"].astype(bool)]
        by.append(dict(
            rule_ensemble=str(ens), mutation_rate=float(mu), w=int(w), n=int(len(g)),
            valid_fraction=float(g["valid_square"].mean()),
            nontrivial_path_fraction=float(gv["nontrivial_path_holonomy"].mean()) if len(gv) else 0.0,
            mean_source_liveness=float(g["source_liveness"].mean()),
            mean_min_edge_transport=float(g["min_edge_transport"].mean()),
            delta_type_counts={str(k): int(v) for k, v in Counter(gv.get("delta_type", pd.Series(dtype=str))).items()},
            group_name_counts={str(k): int(v) for k, v in Counter(gv.get("generated_group_name", pd.Series(dtype=str))).items()},
            gauge_covariance_success=float(gv["gauge_covariance_success"].dropna().mean()) if len(gv) else math.nan,
            conjugacy_class_success=float(gv["conjugacy_class_success"].dropna().mean()) if len(gv) else math.nan,
        ))
    perm0 = [r for r in by if r["rule_ensemble"] == "permutive" and abs(r["mutation_rate"]) < 1e-12]
    copy0 = [r for r in by if r["rule_ensemble"] == "copy" and abs(r["mutation_rate"]) < 1e-12]
    rand0 = [r for r in by if r["rule_ensemble"] == "random" and abs(r["mutation_rate"]) < 1e-12]
    if perm0 and perm0[0]["valid_fraction"] > 0.8 and perm0[0]["nontrivial_path_fraction"] > 0.5 and (not math.isnan(cov)) and cov > 0.99 and conj > 0.99:
        verdict = "SHARED-CORNER FINITE HOLONOMY: one microscopic square has gauge-covariant path dependence"
    elif valid_frac > 0 and nontriv_frac > 0:
        verdict = "PARTIAL SHARED-CORNER HOLONOMY SIGNAL"
    elif valid_frac > 0:
        verdict = "FLAT SHARED-CORNER TRANSPORT: valid squares but no nontrivial path holonomy"
    else:
        verdict = "NO VALID SHARED-CORNER TRANSPORT SQUARES"
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        valid_square_fraction=valid_frac,
        nontrivial_path_holonomy_fraction=nontriv_frac,
        gauge_covariance_success=cov,
        conjugacy_class_success=conj,
        delta_type_counts={str(k): int(v) for k, v in Counter(valid.get("delta_type", pd.Series(dtype=str))).items()},
        generated_group_counts={str(k): int(v) for k, v in Counter(valid.get("generated_group_name", pd.Series(dtype=str))).items()},
        by_ensemble=by,
    )


def plot_shared_square(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    g = df.groupby(["rule_ensemble", "mutation_rate"])
    for ens in sorted(df["rule_ensemble"].unique()):
        sub = []
        for mu in sorted(df["mutation_rate"].unique()):
            gg = df[(df.rule_ensemble == ens) & (df.mutation_rate == mu)]
            gv = gg[gg.valid_square.astype(bool)]
            sub.append((mu, float(gv.nontrivial_path_holonomy.mean()) if len(gv) else 0.0))
        ax[0].plot([x for x, _ in sub], [y for _, y in sub], "o-", label=str(ens))
    ax[0].set_title("shared-corner path dependence")
    ax[0].set_xlabel("mutation rate"); ax[0].set_ylabel("nontrivial path fraction")
    ax[0].legend(fontsize=8)
    for ens in sorted(df["rule_ensemble"].unique()):
        sub = []
        for mu in sorted(df["mutation_rate"].unique()):
            gg = df[(df.rule_ensemble == ens) & (df.mutation_rate == mu)]
            sub.append((mu, float(gg.valid_square.mean()) if len(gg) else 0.0))
        ax[1].plot([x for x, _ in sub], [y for _, y in sub], "o-", label=str(ens))
    ax[1].set_title("valid shared-corner square")
    ax[1].set_xlabel("mutation rate"); ax[1].set_ylabel("valid fraction")
    vc = df[df.valid_square.astype(bool)]
    counts = Counter(vc.delta_type.tolist()) if len(vc) else Counter()
    ax[2].bar(list(counts.keys()), list(counts.values()))
    ax[2].set_title("path holonomy type")
    ax[2].tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _parse_list(s: str, typ=str):
    if not s:
        return []
    return [typ(x) for x in str(s).split(",") if str(x).strip()]


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fully shared-corner square holonomy test.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--ws", default="1")
    p.add_argument("--instances", type=int, default=50)
    p.add_argument("--ensembles", default="copy,permutive,block,canalizing,random")
    p.add_argument("--mutation-rates", default="0,0.02,0.05")
    p.add_argument("--initial-mode", choices=["source_all", "source_random", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", default="example_results/shared_square_holonomy.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    df = run_shared_square_sweep(
        q=int(args.q),
        ws=_parse_list(args.ws, int),
        ensembles=_parse_list(args.ensembles, str),
        mutation_rates=_parse_list(args.mutation_rates, float),
        instances=int(args.instances),
        initial_mode=args.initial_mode,
        n_random_initial=int(args.n_random_initial),
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    summary = analyze_shared_square(df)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_shared_square(df, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
