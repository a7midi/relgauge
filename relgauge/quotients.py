"""
quotients.py -- the three quotients of a single-SCC relational system.

(1) partition_quotient  : the full-state consistency-forcing least fixed point
    Lambda* (treat the schedule as *strict* gauge).  Provably collapses pure
    cycles on the recurrent set -- the "too aggressive" object.

(2) orbit dimension     : log|reachable| / log|X| under the nondeterministic
    lift (track *all* schedule outcomes).  ~ (k-1)/k on cycles -- the
    "too generous" object.

(3) boundary bisimulation: observational equivalence of states as seen through
    a width-w boundary, under the schedule-gauge.  This is the carrier of the
    resolvability quantity R (observables.py).

All exact, finite, deterministic.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import core as C


# --------------------------------------------------------------------------- #
# union-find
# --------------------------------------------------------------------------- #
class UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True

    def classes(self, items):
        d = {}
        for x in items:
            d.setdefault(self.find(x), []).append(x)
        return list(d.values())


# --------------------------------------------------------------------------- #
# (1) partition quotient  Lambda*
# --------------------------------------------------------------------------- #
def partition_quotient(sys: C.RelationalSystem,
                       step_maps: np.ndarray | None = None):
    """Least fixed point of the consistency-forcing operator.

    Per vertex v we maintain an equivalence on [q].  We close under:
      schedule forcing : for every state x, all values {T_sigma(x)[v]} are merged
                         (the whole orbit of v's output value at x is one class);
      admissibility    : if all predecessor values of two input tuples are
                         currently equivalent, their outputs are merged.
    Returns: list of length-q class-label arrays, one per vertex.
    """
    k, q = sys.k, sys.q
    if step_maps is None:
        step_maps = sys.all_step_maps()
    S = C.all_states(k, q)
    # decode successor *state arrays* per schedule once (vertex-wise values)
    # succ_vals[s] is (M,k): the value of each vertex after schedule s
    M = q ** k
    powers = q ** np.arange(k, dtype=np.int64)
    succ_vals = np.empty((step_maps.shape[0], M, k), dtype=np.int64)
    for s in range(step_maps.shape[0]):
        idx = step_maps[s]
        for v in range(k):
            succ_vals[s, :, v] = (idx // (q ** v)) % q

    uf = [UF(q) for _ in range(k)]

    # ---- schedule forcing: connect value-under-sched-0 with value-under-sched-s
    # for every state; transitively merges every value in each state's output
    # orbit.  Only unique (a,b) pairs per (v,s) matter (<= q^2).
    base = succ_vals[0]                       # (M,k)
    for s in range(1, succ_vals.shape[0]):
        cur = succ_vals[s]
        for v in range(k):
            pairs = np.unique(np.stack([base[:, v], cur[:, v]], axis=1), axis=0)
            for a, b in pairs:
                uf[v].union(int(a), int(b))

    # ---- admissibility forcing to fixed point
    # for each vertex, group input tuples by their current class-signature and
    # merge the outputs within each group; iterate until stable.
    def label_of(v, val):
        return uf[v].find(val)

    changed = True
    # precompute input tuples per vertex
    input_tuples = []
    for v in range(k):
        d = len(sys.preds[v])
        if d == 0:
            input_tuples.append(np.zeros((1, 0), dtype=np.int64))
        else:
            input_tuples.append(C.all_states(d, q))  # (q^d, d) over predecessors
    while changed:
        changed = False
        for v in range(k):
            pv = sys.preds[v]
            d = len(pv)
            if d == 0:
                continue
            tuples = input_tuples[v]
            # signature: class-labels of predecessor values
            sigs = {}
            for row_idx in range(tuples.shape[0]):
                vals = tuples[row_idx]
                sig = tuple(label_of(pv[j], int(vals[j])) for j in range(d))
                out = int(sys.rules[v][row_idx])
                sigs.setdefault(sig, []).append(out)
            for outs in sigs.values():
                first = outs[0]
                for o in outs[1:]:
                    if uf[v].union(first, o):
                        changed = True

    labels = [np.array([uf[v].find(a) for a in range(q)], dtype=np.int64)
              for v in range(k)]
    return labels


def partition_class_count(labels, vertex: int, restrict_values=None) -> int:
    """Number of distinct classes at `vertex`, optionally restricted to a set
    of values (e.g. the image/recurrent projection)."""
    lab = labels[vertex]
    if restrict_values is None:
        vals = range(len(lab))
    else:
        vals = restrict_values
    return len({int(lab[v]) for v in vals})


# --------------------------------------------------------------------------- #
# (2) orbit dimension
# --------------------------------------------------------------------------- #
def orbit_dimension(sys: C.RelationalSystem, step_maps: np.ndarray | None = None,
                    which="image"):
    """log_q(|set|) / k  for the reachable-after-one-step set ('image') or the
    recurrent set ('recurrent') of the nondeterministic lift.  -> (k-1)/k on
    cycles for 'image' as q grows."""
    if step_maps is None:
        step_maps = sys.all_step_maps()
    if which == "image":
        size = len(C.image_states(step_maps))
    else:
        adj = C.orbit_adjacency_fast(step_maps)
        size = len(C.recurrent_states(adj))
    if size <= 1:
        return 0.0
    return np.log(size) / np.log(sys.q ** sys.k)


# --------------------------------------------------------------------------- #
# (3) boundary bisimulation  (carrier of resolvability)
# --------------------------------------------------------------------------- #
def _boundary_proj(state_index: np.ndarray, k: int, q: int,
                   boundary: tuple[int, ...]) -> np.ndarray:
    """Project state indices onto the boundary vertices -> a small integer code."""
    if len(boundary) == 0:
        return np.zeros_like(state_index)
    digits = np.empty((len(state_index), len(boundary)), dtype=np.int64)
    for j, v in enumerate(boundary):
        digits[:, j] = (state_index // (q ** v)) % q
    bpow = q ** np.arange(len(boundary), dtype=np.int64)
    return (digits * bpow).sum(axis=1)


def boundary_bisimulation(sys: C.RelationalSystem, boundary: tuple[int, ...],
                          step_maps: np.ndarray | None = None) -> np.ndarray:
    """Largest bisimulation on the boundary-labeled nondeterministic LTS.

    States colored by their boundary projection; transition x -> y carries
    observation = boundary projection of y.  Returns a length-(q**k) array of
    block ids (the observational-equivalence class of each state).
    """
    k, q = sys.k, sys.q
    if step_maps is None:
        step_maps = sys.all_step_maps()
    M = q ** k
    state_idx = np.arange(M, dtype=np.int64)
    bcode = _boundary_proj(state_idx, k, q, boundary)        # color of each state

    # successor sets with observation labels:  succ[x] = set of (obs, y)
    # build from unique edges
    src = np.tile(np.arange(M, dtype=np.int64), step_maps.shape[0])
    dst = step_maps.reshape(-1)
    key = np.unique(src * M + dst)
    e_src = (key // M).astype(np.int64)
    e_dst = (key % M).astype(np.int64)
    e_obs = bcode[e_dst]
    succ = [[] for _ in range(M)]
    for s, o, d in zip(e_src.tolist(), e_obs.tolist(), e_dst.tolist()):
        succ[s].append((o, d))

    # partition refinement (Kanellakis-Smolka): refine by (color,
    # set of (obs, block_of_successor)) until stable.
    block = bcode.copy()                                     # initial: by color
    # compress to 0..b-1
    _, block = np.unique(block, return_inverse=True)
    block = block.astype(np.int64)
    while True:
        sig_to_id = {}
        new_block = np.empty(M, dtype=np.int64)
        for x in range(M):
            sig = (int(block[x]),
                   frozenset((o, int(block[d])) for (o, d) in succ[x]))
            if sig not in sig_to_id:
                sig_to_id[sig] = len(sig_to_id)
            new_block[x] = sig_to_id[sig]
        if len(sig_to_id) == len(set(block.tolist())):
            block = new_block
            break
        block = new_block
    return block


def passive_signature_classes(sys: C.RelationalSystem, boundary: tuple[int, ...],
                              horizon: int = 1,
                              step_maps: np.ndarray | None = None) -> np.ndarray:
    """Conservative *passive* resolver: classes by the set of boundary
    observation-sequences of length `horizon` reachable under the lift.
    horizon=1 is the one-step orbit signature; larger -> trace equivalence."""
    k, q = sys.k, sys.q
    if step_maps is None:
        step_maps = sys.all_step_maps()
    M = q ** k
    bcode = _boundary_proj(np.arange(M), k, q, boundary)
    adj = C.orbit_adjacency_fast(step_maps)

    # signature[x] = frozenset of observation tuples of length `horizon`
    def obs_seqs(x, h):
        if h == 0:
            return {()}
        out = set()
        for y in adj[x]:
            for tail in obs_seqs(y, h - 1):
                out.add((int(bcode[y]),) + tail)
        return out

    sigs = {}
    classes = np.empty(M, dtype=np.int64)
    cache = {}

    def cached(x, h):
        key = (x, h)
        if key not in cache:
            cache[key] = frozenset(obs_seqs(x, h))
        return cache[key]

    for x in range(M):
        sig = (int(bcode[x]), cached(x, horizon))
        if sig not in sigs:
            sigs[sig] = len(sigs)
        classes[x] = sigs[sig]
    return classes
