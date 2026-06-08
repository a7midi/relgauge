"""
core.py -- finite relational dynamical systems and their schedule dynamics.

A RelationalSystem is a finite directed multigraph with finite local state
spaces and deterministic local update rules.  We work with a *single strongly
connected component* (one "observer") for the resolvability experiments: every
vertex's predecessors lie inside the SCC, so all k! vertex orderings are
admissible schedules.

State encoding: a global state (x_0,...,x_{k-1}) in [q]^k is encoded as the
integer  sum_v x_v * q**v.  All states are enumerated as a (q**k, k) int array.

No claims are made here; this module is pure mechanism.  Everything is exact
finite computation.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# state-index helpers
# --------------------------------------------------------------------------- #
def all_states(k: int, q: int) -> np.ndarray:
    """Return the (q**k, k) array of all states; row i is the base-q digits of i
    with digit v in the q**v place (so encode(all_states) == arange)."""
    idx = np.arange(q ** k, dtype=np.int64)
    S = np.empty((q ** k, k), dtype=np.int64)
    for v in range(k):
        S[:, v] = (idx // (q ** v)) % q
    return S


def encode(S: np.ndarray, q: int) -> np.ndarray:
    """Encode a (M,k) state array to length-M index array."""
    k = S.shape[1]
    powers = q ** np.arange(k, dtype=np.int64)
    return (S * powers).sum(axis=1)


# --------------------------------------------------------------------------- #
# the system
# --------------------------------------------------------------------------- #
@dataclass
class RelationalSystem:
    """A finite relational dynamical system on k vertices over alphabet [q].

    preds[v]      : tuple of predecessor vertices (the inputs read by vertex v),
                    in a fixed order used to index the rule table.
    rules[v]      : 1-D int array of length q**len(preds[v]); rules[v][idx] is the
                    output value, where idx = sum_j vals[j]*q**j over predecessor
                    values vals (in preds[v] order).
    """
    k: int
    q: int
    preds: list[tuple[int, ...]]
    rules: list[np.ndarray]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.preds) == self.k
        assert len(self.rules) == self.k
        for v in range(self.k):
            d = len(self.preds[v])
            assert self.rules[v].shape == (self.q ** d,), (
                f"rule {v} has wrong size")

    # ---- single-vertex / schedule updates -------------------------------- #
    def _input_index(self, S: np.ndarray, v: int) -> np.ndarray:
        """Vectorized predecessor-tuple index for every row of state array S."""
        pv = self.preds[v]
        if len(pv) == 0:
            return np.zeros(S.shape[0], dtype=np.int64)
        powers = self.q ** np.arange(len(pv), dtype=np.int64)
        return (S[:, list(pv)] * powers).sum(axis=1)

    def step_map(self, schedule: tuple[int, ...]) -> np.ndarray:
        """Return length-(q**k) array: succ[i] = encode(T_schedule(state_i)).

        Sequential in-place semantics: updating v reads the *current* (possibly
        already-updated this tick) predecessor values.
        """
        S = all_states(self.k, self.q)            # (M,k) working copy
        for v in schedule:
            if len(self.preds[v]) == 0:
                continue                          # source: identity (won't occur in SCC)
            idx = self._input_index(S, v)
            S[:, v] = self.rules[v][idx]
        return encode(S, self.q)

    def all_step_maps(self) -> np.ndarray:
        """(n_sched, q**k) array of successor indices, one row per schedule.
        Builds the state array once and reuses it (vectorized)."""
        scheds = list(itertools.permutations(range(self.k)))
        M = self.q ** self.k
        out = np.empty((len(scheds), M), dtype=np.int64)
        base = all_states(self.k, self.q)
        for s, sched in enumerate(scheds):
            S = base.copy()
            for v in sched:
                if len(self.preds[v]) == 0:
                    continue
                S[:, v] = self.rules[v][self._input_index(S, v)]
            out[s] = encode(S, self.q)
        return out


# --------------------------------------------------------------------------- #
# ensembles  (random instances)
# --------------------------------------------------------------------------- #
def _random_rule(q: int, indeg: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, q, size=q ** indeg, dtype=np.int64)


def make_cycle(k: int, q: int, rng: np.random.Generator) -> RelationalSystem:
    """Directed k-cycle 0->1->...->(k-1)->0 with random single-input rules.
    The sparsest strongly connected graph (in-degree 1 everywhere)."""
    preds = [((v - 1) % k,) for v in range(k)]
    rules = [_random_rule(q, 1, rng) for _ in range(k)]
    return RelationalSystem(k, q, preds, rules, meta=dict(ensemble="cycle"))


def make_random_scc(k: int, q: int, extra: int,
                    rng: np.random.Generator) -> RelationalSystem:
    """Cycle backbone (guarantees strong connectivity) plus `extra` random
    distinct internal edges.  `extra` tunes feedback density: extra=0 -> cycle."""
    pred_sets = [set([(v - 1) % k]) for v in range(k)]   # backbone
    possible = [(u, v) for u in range(k) for v in range(k)
                if u != v and u != (v - 1) % k]
    rng.shuffle(possible)
    for (u, v) in possible[:max(0, extra)]:
        pred_sets[v].add(u)
    preds = [tuple(sorted(pred_sets[v])) for v in range(k)]
    rules = [_random_rule(q, len(preds[v]), rng) for v in range(k)]
    return RelationalSystem(k, q, preds, rules,
                            meta=dict(ensemble=f"scc_extra{extra}"))


def make_regular_scc(k: int, q: int, indeg: int,
                     rng: np.random.Generator) -> RelationalSystem:
    """Each vertex has in-degree `indeg`; cycle backbone ensures connectivity."""
    indeg = max(1, min(indeg, k - 1))
    pred_sets = [set([(v - 1) % k]) for v in range(k)]
    for v in range(k):
        cand = [u for u in range(k) if u != v and u not in pred_sets[v]]
        rng.shuffle(cand)
        for u in cand[: indeg - len(pred_sets[v])]:
            pred_sets[v].add(u)
    preds = [tuple(sorted(pred_sets[v])) for v in range(k)]
    rules = [_random_rule(q, len(preds[v]), rng) for v in range(k)]
    return RelationalSystem(k, q, preds, rules,
                            meta=dict(ensemble=f"regular_d{indeg}"))


ENSEMBLES = {
    "cycle": lambda k, q, rng, **kw: make_cycle(k, q, rng),
    "scc": lambda k, q, rng, extra=2, **kw: make_random_scc(k, q, extra, rng),
    "regular": lambda k, q, rng, indeg=2, **kw: make_regular_scc(k, q, indeg, rng),
}


# --------------------------------------------------------------------------- #
# graph / SCC utilities (iterative Tarjan, safe for large vertex counts)
# --------------------------------------------------------------------------- #
def tarjan_scc(adj: dict[int, set[int]], nodes) -> list[list[int]]:
    """Strongly connected components of a directed graph given as adjacency
    sets.  Iterative to avoid recursion limits.  Returns list of components."""
    index = {}
    low = {}
    onstack = {}
    stack = []
    comps = []
    counter = 0
    nodes = list(nodes)
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        onstack[root] = True
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    onstack[w] = True
                    work.append((w, iter(adj.get(w, ()))))
                    advanced = True
                    break
                elif onstack.get(w, False):
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    onstack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(comp)
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
    return comps


def condensation_is_dag(adj: dict[int, set[int]], nodes) -> bool:
    """Structural check S1: the condensation (SCC quotient) is acyclic."""
    comps = tarjan_scc(adj, nodes)
    comp_id = {}
    for i, c in enumerate(comps):
        for v in c:
            comp_id[v] = i
    cadj = {i: set() for i in range(len(comps))}
    for u in nodes:
        for w in adj.get(u, ()):
            if comp_id[u] != comp_id[w]:
                cadj[comp_id[u]].add(comp_id[w])
    # acyclic iff its own SCCs are all singletons
    ccomps = tarjan_scc(cadj, range(len(comps)))
    return all(len(c) == 1 for c in ccomps)


# --------------------------------------------------------------------------- #
# orbit (nondeterministic) dynamics: reachable / recurrent / image
# --------------------------------------------------------------------------- #
def orbit_adjacency(step_maps: np.ndarray) -> dict[int, set[int]]:
    """state -> set of one-step successors over all schedules (the lift N)."""
    M = step_maps.shape[1]
    adj = {i: set() for i in range(M)}
    for row in step_maps:
        for src in range(M):
            adj[src].add(int(row[src]))
    return adj


def orbit_adjacency_fast(step_maps: np.ndarray) -> dict[int, set[int]]:
    """Vectorized: collect unique (src,dst) edges across schedules."""
    M = step_maps.shape[1]
    src = np.tile(np.arange(M, dtype=np.int64), step_maps.shape[0])
    dst = step_maps.reshape(-1)
    key = src * M + dst
    key = np.unique(key)
    src_u = key // M
    dst_u = key % M
    adj = {i: set() for i in range(M)}
    for s, d in zip(src_u.tolist(), dst_u.tolist()):
        adj[s].add(d)
    return adj


def recurrent_states(adj: dict[int, set[int]]) -> np.ndarray:
    """States on a directed cycle: members of an SCC of size>1, or with a
    self-loop.  These are the dynamically recurrent states of the lift."""
    nodes = list(adj.keys())
    comps = tarjan_scc(adj, nodes)
    rec = []
    for c in comps:
        if len(c) > 1:
            rec.extend(c)
        else:
            v = c[0]
            if v in adj.get(v, ()):
                rec.append(v)
    return np.array(sorted(rec), dtype=np.int64)


def image_states(step_maps: np.ndarray) -> np.ndarray:
    """States reachable in exactly one step under some schedule."""
    return np.unique(step_maps.reshape(-1))


def synchronous_step_map(sys) -> np.ndarray:
    """Synchronous one-tick map: every vertex reads OLD values simultaneously.
    Under this semantics information crosses exactly one edge per tick (the
    finite-light-cone claim).  Contrast with schedule (sequential) dynamics,
    where a single tick can propagate along a whole path."""
    k, q = sys.k, sys.q
    S = all_states(k, q)
    out = np.empty_like(S)
    for v in range(k):
        if len(sys.preds[v]) == 0:
            out[:, v] = S[:, v]
        else:
            out[:, v] = sys.rules[v][sys._input_index(S, v)]
    return encode(out, q)
