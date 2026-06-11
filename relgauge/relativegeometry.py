"""
relativegeometry.py -- observer-relative boundary geometry in closed finite systems.

The experiment implemented here is the direct test of the principle:

    no exterior -> no absolute boundary -> geometry is observer-relative.

Given a closed finite RelationalSystem, the full vertex set has empty boundary.
Every proper subsystem O can have a boundary, but only the distinctions that O's
finite internal state can resolve through its live interface are treated as
physical/effective boundary data.

The module deliberately avoids hand-picked plaquettes or lattices.  It:

  1. enumerates small proper candidate observers O;
  2. computes their raw cut boundary inside the closed graph;
  3. computes an exact finite-memory boundary channel
         exterior boundary word at t=0 -> possible O memory at horizon T
     with schedule order treated as gauge/noise;
  4. quotients boundary words by zero-error confusability of O's final memory;
  5. extracts a minimal observer-relative boundary geometry from live resolved
     boundary sites; and
  6. audits whether resolved boundaries cancel/glue correctly under observer
     union A ∪ B.  Nonzero H(Q_{A∪B} | Q_A, Q_B) is the first residual/defect
     diagnostic: local observer boundary data do not determine the union's
     boundary quotient.

All computations are exact for the enumerated finite state space and supplied
schedule sample.  Use small k/q or sampled schedules for exploratory sweeps.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import core as C
from . import finiteobserver as F


# --------------------------------------------------------------------------- #
# Basic information utilities
# --------------------------------------------------------------------------- #


def _entropy_from_counts(counts: Iterable[int | float]) -> float:
    vals = np.asarray([float(c) for c in counts if float(c) > 0.0], dtype=float)
    if vals.size == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-np.sum(p * np.log2(p)))


def _mutual_information_from_channel_uniform(P: np.ndarray) -> float:
    """I(X;Y) in bits for a channel P[y|x] with uniform prior on rows."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] == 0:
        return 0.0
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    P = P / row_sum
    px = np.full(P.shape[0], 1.0 / P.shape[0])
    py = px @ P
    eps = 1e-300
    mask = P > 0
    log_ratio = np.zeros_like(P)
    log_ratio[mask] = np.log2(P[mask] + eps) - np.log2(np.broadcast_to(py, P.shape)[mask] + eps)
    return max(0.0, float(np.sum(px[:, None] * P * log_ratio)))


def _digits(indices: np.ndarray, q: int, vertices: Sequence[int]) -> np.ndarray:
    """Return base-q digits for selected global vertices."""
    idx = np.asarray(indices, dtype=np.int64)
    if len(vertices) == 0:
        return np.zeros((len(idx), 0), dtype=np.int64)
    out = np.empty((len(idx), len(vertices)), dtype=np.int64)
    for j, v in enumerate(vertices):
        out[:, j] = (idx // (q ** int(v))) % q
    return out


def _encode_local_digits(D: np.ndarray, q: int) -> np.ndarray:
    """Encode rows of a local digit matrix using local base-q order."""
    D = np.asarray(D, dtype=np.int64)
    if D.ndim != 2:
        raise ValueError("D must be a matrix")
    if D.shape[1] == 0:
        return np.zeros(D.shape[0], dtype=np.int64)
    powers = q ** np.arange(D.shape[1], dtype=np.int64)
    return (D * powers).sum(axis=1).astype(np.int64)


def _as_tuple(vertices: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in vertices))


# --------------------------------------------------------------------------- #
# Closed-system ensembles
# --------------------------------------------------------------------------- #


def _identity_rule(q: int) -> np.ndarray:
    return np.arange(q, dtype=np.int64)


def _constant_rule(q: int, indeg: int, value: int = 0) -> np.ndarray:
    return np.full(q ** indeg, int(value) % q, dtype=np.int64)


def _xor_rule(indeg: int) -> np.ndarray:
    """q=2 parity rule on indeg inputs."""
    vals = np.arange(2 ** indeg, dtype=np.int64)
    out = np.zeros_like(vals)
    for j in range(indeg):
        out ^= (vals >> j) & 1
    return out


def make_closed_system(
    k: int,
    q: int,
    rng: np.random.Generator,
    ensemble: str = "random_scc",
    extra: int | None = None,
    indeg: int = 2,
) -> C.RelationalSystem:
    """Small closed-system ensembles for observer-relative geometry tests.

    Ensembles
    ---------
    random_scc:
        Cycle backbone plus random extra edges and random deterministic rules.
    regular:
        Random regular-ish SCC with fixed in-degree and random rules.
    copy_cycle:
        Directed ring, each vertex copies its predecessor.
    observer_copy_pair:
        Closed two-region positive control: a stable exterior memory symbol feeds
        a recurrent observer SCC, so the observer has a live resolved boundary.
    observer_parity_pair:
        Same closed two-region control, but the observer only receives the
        exterior symbol modulo 2.  This is a q=4-friendly C2/binary control.
    permutation_cycle:
        Directed ring, each edge carries an independently sampled q-symbol
        permutation.  A reversible single-input control.
    xor_ring:
        q=2, each vertex updates to XOR of its two ring neighbors.
    constant:
        Directed ring with constant rules.  Negative control.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    if q < 2:
        raise ValueError("q must be at least 2")
    ens = str(ensemble).strip().lower()
    if extra is None:
        extra = k
    if ens in {"random", "random_scc", "scc"}:
        return C.make_random_scc(k, q, int(extra), rng)
    if ens == "regular":
        return C.make_regular_scc(k, q, int(indeg), rng)
    if ens == "copy_cycle":
        preds = [((v - 1) % k,) for v in range(k)]
        rules = [_identity_rule(q) for _ in range(k)]
        return C.RelationalSystem(k, q, preds, rules, meta={"ensemble": ens})
    if ens in {"observer_copy_pair", "observer_parity_pair"}:
        if k < 4:
            raise ValueError(f"{ens} requires k >= 4")
        kB = max(1, k // 2)
        kA = k - kB
        if kA < 2:
            raise ValueError(f"{ens} requires at least two observer vertices")
        preds: list[tuple[int, ...]] = []
        rules: list[np.ndarray] = []
        # Stable exterior region B: each B vertex is a self-copy memory cell.
        for v in range(kB):
            preds.append((v,))
            rules.append(_identity_rule(q))
        # Recurrent observer region A.  A0 has an internal predecessor A_last
        # and an exterior predecessor B0.  For observer_copy_pair it copies the
        # full B0 symbol.  For observer_parity_pair it copies only B0 mod 2,
        # providing a q=4-friendly binary/C2 positive control without a lattice.
        a0 = kB
        alast = k - 1
        preds.append((0, alast))
        r0 = np.empty(q ** 2, dtype=np.int64)
        for idx in range(q ** 2):
            b0 = idx % q
            r0[idx] = b0 if ens == "observer_copy_pair" else (b0 % 2)
        rules.append(r0)
        for a in range(1, kA):
            preds.append((kB + a - 1,))
            rules.append(_identity_rule(q))
        return C.RelationalSystem(k, q, preds, rules, meta={"ensemble": ens, "kB": kB, "kA": kA})
    if ens == "permutation_cycle":
        preds = [((v - 1) % k,) for v in range(k)]
        rules = []
        for _ in range(k):
            p = np.arange(q, dtype=np.int64)
            rng.shuffle(p)
            rules.append(p)
        return C.RelationalSystem(k, q, preds, rules, meta={"ensemble": ens})
    if ens == "xor_ring":
        if q != 2:
            raise ValueError("xor_ring is defined only for q=2")
        preds = [tuple(sorted({(v - 1) % k, (v + 1) % k})) for v in range(k)]
        rules = [_xor_rule(len(preds[v])) for v in range(k)]
        return C.RelationalSystem(k, q, preds, rules, meta={"ensemble": ens})
    if ens == "constant":
        preds = [((v - 1) % k,) for v in range(k)]
        rules = [_constant_rule(q, 1, 0) for _ in range(k)]
        return C.RelationalSystem(k, q, preds, rules, meta={"ensemble": ens})
    raise ValueError(f"unknown closed-system ensemble: {ensemble!r}")


# --------------------------------------------------------------------------- #
# Graph/boundary utilities
# --------------------------------------------------------------------------- #


def edge_adjacency(sys: C.RelationalSystem) -> dict[int, set[int]]:
    """Directed adjacency u -> v induced by predecessor lists."""
    adj = {v: set() for v in range(sys.k)}
    for v in range(sys.k):
        for u in sys.preds[v]:
            adj[int(u)].add(v)
    return adj


def directed_edges(sys: C.RelationalSystem) -> set[tuple[int, int]]:
    return {(int(u), v) for v in range(sys.k) for u in sys.preds[v]}


def boundary_edges(sys: C.RelationalSystem, observer: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Raw directed cut boundary ∂O: edges with exactly one endpoint in O."""
    O = set(int(v) for v in observer)
    if not O:
        return tuple()
    out = []
    for u, v in directed_edges(sys):
        if (u in O) ^ (v in O):
            out.append((u, v))
    return tuple(sorted(out))


def incoming_boundary_edges(sys: C.RelationalSystem, observer: Iterable[int]) -> tuple[tuple[int, int], ...]:
    O = set(int(v) for v in observer)
    return tuple(sorted((u, v) for (u, v) in boundary_edges(sys, O) if v in O and u not in O))


def outgoing_boundary_edges(sys: C.RelationalSystem, observer: Iterable[int]) -> tuple[tuple[int, int], ...]:
    O = set(int(v) for v in observer)
    return tuple(sorted((u, v) for (u, v) in boundary_edges(sys, O) if u in O and v not in O))


def exterior_boundary_support(sys: C.RelationalSystem, observer: Iterable[int]) -> tuple[int, ...]:
    """Complement-side vertices incident to the cut."""
    O = set(int(v) for v in observer)
    support = set()
    for u, v in boundary_edges(sys, O):
        support.add(v if u in O else u)
    return tuple(sorted(support))


def interior_boundary_support(sys: C.RelationalSystem, observer: Iterable[int]) -> tuple[int, ...]:
    """Observer-side vertices incident to the cut."""
    O = set(int(v) for v in observer)
    support = set()
    for u, v in boundary_edges(sys, O):
        support.add(u if u in O else v)
    return tuple(sorted(support))


def internal_feedback(sys: C.RelationalSystem, observer: Iterable[int]) -> bool:
    """True if the induced subgraph on O has an SCC of size>1 or a self-loop."""
    O = set(int(v) for v in observer)
    if len(O) < 1:
        return False
    adj_full = edge_adjacency(sys)
    adj = {v: {w for w in adj_full.get(v, set()) if w in O} for v in O}
    comps = C.tarjan_scc(adj, O)
    if any(len(c) > 1 for c in comps):
        return True
    return any(v in adj.get(v, set()) for v in O)


def is_proper_observer_candidate(
    sys: C.RelationalSystem,
    observer: Iterable[int],
    require_feedback: bool = True,
    require_boundary: bool = True,
) -> bool:
    O = set(int(v) for v in observer)
    if not O or len(O) >= sys.k:
        return False
    if require_boundary and not boundary_edges(sys, O):
        return False
    if require_feedback and not internal_feedback(sys, O):
        return False
    return True


def enumerate_observer_candidates(
    sys: C.RelationalSystem,
    min_size: int = 2,
    max_size: int | None = None,
    require_feedback: bool = True,
    max_observers: int | None = None,
) -> list[tuple[int, ...]]:
    """Enumerate small proper observer subsets.

    This intentionally does not use global SCCs.  In a closed SCC universe the
    whole system may be one SCC, but proper recurrent subsystems can still act
    as internal observers if they have feedback and a live boundary.
    """
    if max_size is None:
        max_size = max(min_size, min(sys.k - 1, 3))
    out: list[tuple[int, ...]] = []
    for r in range(int(min_size), int(max_size) + 1):
        if r <= 0 or r >= sys.k + 1:
            continue
        for combo in itertools.combinations(range(sys.k), r):
            if is_proper_observer_candidate(sys, combo, require_feedback=require_feedback):
                out.append(tuple(combo))
                if max_observers is not None and len(out) >= int(max_observers):
                    return out
    return out


def raw_boundary_cancellation_holds(
    sys: C.RelationalSystem,
    A: Iterable[int],
    B: Iterable[int],
) -> bool:
    """Check ∂(A∪B) = ∂A Δ ∂B for disjoint A,B at raw directed-edge level."""
    A = set(int(v) for v in A)
    B = set(int(v) for v in B)
    if A & B:
        raise ValueError("raw boundary cancellation currently requires disjoint A and B")
    left = set(boundary_edges(sys, A)) ^ set(boundary_edges(sys, B))
    right = set(boundary_edges(sys, A | B))
    return left == right


def seam_edges(sys: C.RelationalSystem, A: Iterable[int], B: Iterable[int]) -> tuple[tuple[int, int], ...]:
    A = set(int(v) for v in A)
    B = set(int(v) for v in B)
    return tuple(sorted((u, v) for (u, v) in directed_edges(sys) if (u in A and v in B) or (u in B and v in A)))


# --------------------------------------------------------------------------- #
# Schedule/recurrent helpers
# --------------------------------------------------------------------------- #


def schedule_sample(
    k: int,
    rng: np.random.Generator,
    max_schedules: int | None = None,
) -> list[tuple[int, ...]]:
    """All vertex permutations unless max_schedules requests a random subset."""
    total = math.factorial(k)
    if max_schedules is None or int(max_schedules) <= 0 or int(max_schedules) >= total:
        return [tuple(p) for p in itertools.permutations(range(k))]
    max_schedules = int(max_schedules)
    seen: set[tuple[int, ...]] = set()
    while len(seen) < max_schedules:
        p = tuple(int(x) for x in rng.permutation(k).tolist())
        seen.add(p)
    return sorted(seen)


def step_maps_for_schedules(sys: C.RelationalSystem, schedules: Sequence[tuple[int, ...]]) -> np.ndarray:
    out = np.empty((len(schedules), sys.q ** sys.k), dtype=np.int64)
    for i, sched in enumerate(schedules):
        out[i] = sys.step_map(tuple(sched))
    return out


def recurrent_states_from_step_maps(step_maps: np.ndarray) -> np.ndarray:
    adj = C.orbit_adjacency_fast(step_maps)
    return C.recurrent_states(adj)


def _advance_set(states: np.ndarray, step_maps: np.ndarray) -> np.ndarray:
    if len(states) == 0:
        return states.astype(np.int64)
    return np.unique(step_maps[:, states].reshape(-1)).astype(np.int64)


def _advance_distribution(dist: np.ndarray, step_maps: np.ndarray) -> np.ndarray:
    nz = np.flatnonzero(dist > 0)
    if len(nz) == 0:
        return np.zeros_like(dist)
    n_sched = step_maps.shape[0]
    dest = step_maps[:, nz].reshape(-1)
    weights = np.broadcast_to(dist[nz], (n_sched, len(nz))).reshape(-1) / float(n_sched)
    out = np.bincount(dest, weights=weights, minlength=len(dist)).astype(float)
    total = out.sum()
    if total > 0:
        out /= total
    return out


# --------------------------------------------------------------------------- #
# Boundary quotient construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoundaryMeasure:
    observer: tuple[int, ...]
    exterior_support: tuple[int, ...]
    interior_support: tuple[int, ...]
    raw_boundary_edges: tuple[tuple[int, int], ...]
    incoming_edges: tuple[tuple[int, int], ...]
    outgoing_edges: tuple[tuple[int, int], ...]
    recurrent_states: tuple[int, ...]
    labels: tuple[tuple[int, ...], ...]
    possible_outputs: dict[int, frozenset[int]]
    label_to_component: dict[int, int]
    component_count: int
    zero_error_code_size: int
    zero_error_exact: bool
    quotient_entropy_bits: float
    label_entropy_bits: float
    memory_entropy_bits: float
    boundary_mi_bits: float
    signature_classes: int
    singleton_output_fraction: float
    mean_output_set_size: float
    live_sites: tuple[int, ...]
    live_site_fraction: float
    effective_edges: tuple[tuple[int, int], ...]
    effective_cycle_rank: int

    @property
    def nontrivial(self) -> bool:
        return self.component_count >= 2 and self.quotient_entropy_bits > 1e-12

    @property
    def binary_c2_like(self) -> bool:
        return self.component_count == 2 and self.quotient_entropy_bits > 1e-12


def _confusability_components(possible_outputs: dict[int, frozenset[int]]) -> dict[int, int]:
    """Connected components of the overlap/confusability graph on labels."""
    labels = sorted(possible_outputs)
    out_sets = {i: set(possible_outputs[i]) for i in labels}
    adj: dict[int, set[int]] = {i: set() for i in labels}
    for a_i, i in enumerate(labels):
        si = out_sets[i]
        for j in labels[a_i + 1:]:
            if not si.isdisjoint(out_sets[j]):
                adj[i].add(j)
                adj[j].add(i)
    comp: dict[int, int] = {}
    cid = 0
    for i in labels:
        if i in comp:
            continue
        q = deque([i])
        comp[i] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in comp:
                    comp[v] = cid
                    q.append(v)
        cid += 1
    return comp


def _label_indices_for_states(
    sys: C.RelationalSystem,
    states: np.ndarray,
    support: Sequence[int],
    labels: Sequence[tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Boundary word index for every state in `states`."""
    words = [tuple(int(x) for x in row) for row in _digits(states, sys.q, support).tolist()]
    if labels is None:
        labels_t = tuple(sorted(set(words)))
    else:
        labels_t = tuple(labels)
    lookup = {lab: i for i, lab in enumerate(labels_t)}
    idx = np.asarray([lookup[w] for w in words], dtype=np.int64)
    return idx, labels_t


def _observer_output_indices(sys: C.RelationalSystem, states: np.ndarray, observer: Sequence[int]) -> np.ndarray:
    return _encode_local_digits(_digits(states, sys.q, observer), sys.q)


def _stochastic_boundary_matrix(
    sys: C.RelationalSystem,
    observer: Sequence[int],
    support: Sequence[int],
    rec_states: np.ndarray,
    labels: Sequence[tuple[int, ...]],
    step_maps: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """P(observer memory at T | boundary word), uniform over compatible recurrent states and schedules."""
    n_outputs = sys.q ** len(observer)
    P = np.zeros((len(labels), n_outputs), dtype=float)
    if len(labels) == 0:
        return P
    rec_words = [tuple(int(x) for x in row) for row in _digits(rec_states, sys.q, support).tolist()]
    states_by_label: dict[int, list[int]] = {i: [] for i in range(len(labels))}
    label_lookup = {lab: i for i, lab in enumerate(labels)}
    for s, w in zip(rec_states.tolist(), rec_words):
        states_by_label[label_lookup[w]].append(int(s))
    n_full = sys.q ** sys.k
    for li in range(len(labels)):
        compatible = states_by_label.get(li, [])
        if not compatible:
            continue
        dist = np.zeros(n_full, dtype=float)
        dist[np.asarray(compatible, dtype=np.int64)] = 1.0 / float(len(compatible))
        for _ in range(int(horizon)):
            dist = _advance_distribution(dist, step_maps)
        nz = np.flatnonzero(dist > 0)
        out = _observer_output_indices(sys, nz, observer)
        P[li] = np.bincount(out, weights=dist[nz], minlength=n_outputs)
        total = P[li].sum()
        if total > 0:
            P[li] /= total
    return P


def _live_boundary_sites(
    sys: C.RelationalSystem,
    observer: Sequence[int],
    support: Sequence[int],
    rec_states: np.ndarray,
    step_maps: np.ndarray,
    horizon: int,
    min_site_entropy_bits: float = 1e-9,
) -> tuple[int, ...]:
    """A boundary site is live if its value has a nontrivial zero-error quotient."""
    live = []
    for site in support:
        vals = sorted(set(_digits(rec_states, sys.q, [site]).reshape(-1).astype(int).tolist()))
        if len(vals) < 2:
            continue
        # Reuse the general boundary channel with this one site as support.
        labels = tuple((int(v),) for v in vals)
        outputs: dict[int, frozenset[int]] = {}
        rec_site = _digits(rec_states, sys.q, [site]).reshape(-1)
        for li, val in enumerate(vals):
            current = rec_states[rec_site == val]
            for _ in range(int(horizon)):
                current = _advance_set(current, step_maps)
            outs = _observer_output_indices(sys, current, observer)
            outputs[li] = frozenset(int(x) for x in np.unique(outs).tolist())
        comps = _confusability_components(outputs)
        comp_count = len(set(comps.values()))
        if comp_count >= 2:
            # Guard against unreachable/zero-entropy site values in recurrent set.
            counts = Counter(rec_site.astype(int).tolist())
            if _entropy_from_counts(counts.values()) > min_site_entropy_bits:
                live.append(int(site))
    return tuple(live)


def _effective_boundary_edges(
    sys: C.RelationalSystem,
    observer: Sequence[int],
    live_sites: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Minimal resolved interface graph among live exterior boundary sites.

    Two live exterior sites are adjacent in Γ_O when they couple to a common
    observer-side boundary vertex.  This is not a plaquette assumption; it is a
    relation extracted from the cut incidence structure and then filtered by
    finite-memory liveness.
    """
    O = set(observer)
    live = set(live_sites)
    incident_by_inside: dict[int, set[int]] = defaultdict(set)
    for u, v in boundary_edges(sys, O):
        inside = u if u in O else v
        outside = v if u in O else u
        if outside in live:
            incident_by_inside[int(inside)].add(int(outside))
    e = set()
    for sites in incident_by_inside.values():
        for a, b in itertools.combinations(sorted(sites), 2):
            e.add((a, b) if a < b else (b, a))
    return tuple(sorted(e))


def _undirected_cycle_rank(nodes: Sequence[int], edges: Sequence[tuple[int, int]]) -> int:
    nodes = list(dict.fromkeys(int(n) for n in nodes))
    if not nodes:
        return 0
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    E = 0
    for a, b in edges:
        if a == b:
            continue
        if a not in parent or b not in parent:
            continue
        E += 1
        union(a, b)
    comps = len({find(n) for n in nodes})
    return max(0, E - len(nodes) + comps)


def boundary_measure(
    sys: C.RelationalSystem,
    observer: Iterable[int],
    horizon: int = 2,
    step_maps: np.ndarray | None = None,
    recurrent_states: np.ndarray | None = None,
    schedules: Sequence[tuple[int, ...]] | None = None,
    rng: np.random.Generator | None = None,
    max_schedules: int | None = None,
    do_shannon: bool = True,
) -> BoundaryMeasure:
    """Compute the resolved observer-relative boundary quotient for O.

    The boundary word is the tuple of complement-side endpoint values on the raw
    cut.  O's only readout is its own final internal state after `horizon` ticks.
    Schedule freedom is propagated as nondeterministic gauge/noise.
    """
    O = _as_tuple(observer)
    if len(O) >= sys.k:
        # The full closed system has no exterior boundary by definition.
        rec_tuple: tuple[int, ...] = tuple()
        return BoundaryMeasure(
            observer=O,
            exterior_support=tuple(),
            interior_support=tuple(),
            raw_boundary_edges=tuple(),
            incoming_edges=tuple(),
            outgoing_edges=tuple(),
            recurrent_states=rec_tuple,
            labels=(tuple(),),
            possible_outputs={0: frozenset()},
            label_to_component={0: 0},
            component_count=1,
            zero_error_code_size=1,
            zero_error_exact=True,
            quotient_entropy_bits=0.0,
            label_entropy_bits=0.0,
            memory_entropy_bits=0.0,
            boundary_mi_bits=0.0,
            signature_classes=1,
            singleton_output_fraction=0.0,
            mean_output_set_size=0.0,
            live_sites=tuple(),
            live_site_fraction=0.0,
            effective_edges=tuple(),
            effective_cycle_rank=0,
        )
    if rng is None:
        rng = np.random.default_rng(0)
    if schedules is None:
        schedules = schedule_sample(sys.k, rng, max_schedules=max_schedules)
    if step_maps is None:
        step_maps = step_maps_for_schedules(sys, schedules)
    if recurrent_states is None:
        recurrent_states = recurrent_states_from_step_maps(step_maps)
    rec_states = np.asarray(recurrent_states, dtype=np.int64)

    raw = boundary_edges(sys, O)
    inc = incoming_boundary_edges(sys, O)
    out = outgoing_boundary_edges(sys, O)
    ext = exterior_boundary_support(sys, O)
    interior = interior_boundary_support(sys, O)

    if len(rec_states) == 0 or len(ext) == 0:
        labels = (tuple(),)
        possible_outputs = {0: frozenset()}
        comp = {0: 0}
        return BoundaryMeasure(
            observer=O,
            exterior_support=ext,
            interior_support=interior,
            raw_boundary_edges=raw,
            incoming_edges=inc,
            outgoing_edges=out,
            recurrent_states=tuple(int(x) for x in rec_states.tolist()),
            labels=labels,
            possible_outputs=possible_outputs,
            label_to_component=comp,
            component_count=1,
            zero_error_code_size=1,
            zero_error_exact=True,
            quotient_entropy_bits=0.0,
            label_entropy_bits=0.0,
            memory_entropy_bits=0.0,
            boundary_mi_bits=0.0,
            signature_classes=1,
            singleton_output_fraction=0.0,
            mean_output_set_size=0.0,
            live_sites=tuple(),
            live_site_fraction=0.0,
            effective_edges=tuple(),
            effective_cycle_rank=0,
        )

    label_idx, labels = _label_indices_for_states(sys, rec_states, ext)
    possible_outputs: dict[int, frozenset[int]] = {}
    for li, _label in enumerate(labels):
        current = rec_states[label_idx == li]
        for _ in range(int(horizon)):
            current = _advance_set(current, step_maps)
        outs = _observer_output_indices(sys, current, O)
        possible_outputs[li] = frozenset(int(x) for x in np.unique(outs).tolist())

    comp = _confusability_components(possible_outputs)
    comp_count = len(set(comp.values())) if comp else 0
    ze_size, ze_exact = F.zero_error_code_size(possible_outputs)

    comp_counts: Counter[int] = Counter()
    label_counts: Counter[int] = Counter(label_idx.astype(int).tolist())
    for li, n in label_counts.items():
        comp_counts[int(comp[int(li)])] += int(n)
    label_entropy = _entropy_from_counts(label_counts.values())
    quotient_entropy = _entropy_from_counts(comp_counts.values())

    out_sizes = np.asarray([len(possible_outputs[i]) for i in sorted(possible_outputs)], dtype=float)
    singleton_frac = float(np.mean(out_sizes == 1.0)) if out_sizes.size else 0.0
    mean_out = float(np.mean(out_sizes)) if out_sizes.size else 0.0
    signatures = {possible_outputs[i] for i in possible_outputs}
    memory_counts: Counter[int] = Counter()
    # Distribution of observer states on recurrent set, not after channel; this is a memory-size diagnostic.
    rec_obs = _observer_output_indices(sys, rec_states, O)
    memory_counts.update(rec_obs.astype(int).tolist())
    memory_entropy = _entropy_from_counts(memory_counts.values())

    boundary_mi = 0.0
    if do_shannon:
        P = _stochastic_boundary_matrix(sys, O, ext, rec_states, labels, step_maps, int(horizon))
        boundary_mi = _mutual_information_from_channel_uniform(P)

    live_sites = _live_boundary_sites(sys, O, ext, rec_states, step_maps, int(horizon))
    eff_edges = _effective_boundary_edges(sys, O, live_sites)
    cycle_rank = _undirected_cycle_rank(live_sites, eff_edges)

    return BoundaryMeasure(
        observer=O,
        exterior_support=ext,
        interior_support=interior,
        raw_boundary_edges=raw,
        incoming_edges=inc,
        outgoing_edges=out,
        recurrent_states=tuple(int(x) for x in rec_states.tolist()),
        labels=labels,
        possible_outputs=possible_outputs,
        label_to_component={int(k): int(v) for k, v in comp.items()},
        component_count=int(comp_count),
        zero_error_code_size=int(ze_size),
        zero_error_exact=bool(ze_exact),
        quotient_entropy_bits=float(quotient_entropy),
        label_entropy_bits=float(label_entropy),
        memory_entropy_bits=float(memory_entropy),
        boundary_mi_bits=float(boundary_mi),
        signature_classes=int(len(signatures)),
        singleton_output_fraction=float(singleton_frac),
        mean_output_set_size=float(mean_out),
        live_sites=tuple(int(x) for x in live_sites),
        live_site_fraction=float(len(live_sites) / len(ext)) if len(ext) else 0.0,
        effective_edges=eff_edges,
        effective_cycle_rank=int(cycle_rank),
    )


def boundary_measure_row(
    instance: int,
    ensemble: str,
    bm: BoundaryMeasure,
    k: int,
    q: int,
    horizon: int,
    n_schedules: int,
) -> dict:
    return dict(
        instance=int(instance),
        rule_ensemble=str(ensemble),
        k=int(k),
        q=int(q),
        horizon=int(horizon),
        n_schedules=int(n_schedules),
        observer=str(tuple(bm.observer)),
        observer_size=int(len(bm.observer)),
        raw_boundary_size=int(len(bm.raw_boundary_edges)),
        incoming_boundary_size=int(len(bm.incoming_edges)),
        outgoing_boundary_size=int(len(bm.outgoing_edges)),
        exterior_support_size=int(len(bm.exterior_support)),
        interior_support_size=int(len(bm.interior_support)),
        n_recurrent_states=int(len(bm.recurrent_states)),
        boundary_label_classes=int(len(bm.labels)),
        boundary_label_entropy_bits=float(bm.label_entropy_bits),
        boundary_quotient_size=int(bm.component_count),
        boundary_quotient_entropy_bits=float(bm.quotient_entropy_bits),
        boundary_nontrivial=bool(bm.nontrivial),
        boundary_binary_c2_like=bool(bm.binary_c2_like),
        boundary_mi_bits=float(bm.boundary_mi_bits),
        zero_error_code_size=int(bm.zero_error_code_size),
        zero_error_exact=bool(bm.zero_error_exact),
        signature_classes=int(bm.signature_classes),
        singleton_output_fraction=float(bm.singleton_output_fraction),
        mean_output_set_size=float(bm.mean_output_set_size),
        live_site_count=int(len(bm.live_sites)),
        live_site_fraction=float(bm.live_site_fraction),
        effective_edge_count=int(len(bm.effective_edges)),
        effective_cycle_rank=int(bm.effective_cycle_rank),
        observer_vertices=json.dumps(list(bm.observer)),
        exterior_support=json.dumps(list(bm.exterior_support)),
        live_sites=json.dumps(list(bm.live_sites)),
        effective_edges=json.dumps([list(e) for e in bm.effective_edges]),
    )


# --------------------------------------------------------------------------- #
# Observer-union/gluing audit
# --------------------------------------------------------------------------- #


def _component_for_state(
    sys: C.RelationalSystem,
    bm: BoundaryMeasure,
    state: int,
) -> int:
    if len(bm.exterior_support) == 0:
        return 0
    word = tuple(int(x) for x in _digits(np.asarray([state], dtype=np.int64), sys.q, bm.exterior_support)[0].tolist())
    try:
        li = bm.labels.index(word)
    except ValueError:
        # State outside the recurrent support used to construct the channel.
        return -1
    return int(bm.label_to_component[int(li)])


@dataclass(frozen=True)
class GluingMeasure:
    A: tuple[int, ...]
    B: tuple[int, ...]
    U: tuple[int, ...]
    raw_cancellation_ok: bool
    seam_edge_count: int
    recurrent_state_count: int
    local_pair_count: int
    union_class_count: int
    ambiguous_pair_count: int
    gluing_accuracy: float
    residual_entropy_bits: float
    defect_candidate: bool


def gluing_measure(
    sys: C.RelationalSystem,
    A: Iterable[int],
    B: Iterable[int],
    horizon: int = 2,
    step_maps: np.ndarray | None = None,
    recurrent_states: np.ndarray | None = None,
    schedules: Sequence[tuple[int, ...]] | None = None,
    rng: np.random.Generator | None = None,
    max_schedules: int | None = None,
    do_shannon: bool = False,
    cache: dict[tuple[int, ...], BoundaryMeasure] | None = None,
) -> GluingMeasure:
    """Audit resolved-boundary cancellation for A,B -> A∪B.

    Raw cancellation checks the Z2 cut identity ∂(A∪B)=∂A Δ ∂B.  Resolved
    gluing then asks whether the union quotient Q_U is determined by the pair of
    local quotients (Q_A,Q_B) over recurrent states.  Nonzero residual entropy is
    the first obstruction/defect signal.
    """
    A_t = _as_tuple(A)
    B_t = _as_tuple(B)
    if set(A_t) & set(B_t):
        raise ValueError("gluing_measure requires disjoint observers")
    U_t = _as_tuple(set(A_t) | set(B_t))
    if len(U_t) >= sys.k:
        # The union has no exterior.  This is still a valid test of collapse to
        # empty boundary, but not an internal observer with an exterior.
        pass
    if rng is None:
        rng = np.random.default_rng(0)
    if schedules is None:
        schedules = schedule_sample(sys.k, rng, max_schedules=max_schedules)
    if step_maps is None:
        step_maps = step_maps_for_schedules(sys, schedules)
    if recurrent_states is None:
        recurrent_states = recurrent_states_from_step_maps(step_maps)
    rec_states = np.asarray(recurrent_states, dtype=np.int64)
    if cache is None:
        cache = {}

    def get_bm(O: tuple[int, ...]) -> BoundaryMeasure:
        if O not in cache:
            cache[O] = boundary_measure(
                sys, O, horizon=horizon, step_maps=step_maps, recurrent_states=rec_states,
                schedules=schedules, rng=rng, max_schedules=max_schedules, do_shannon=do_shannon,
            )
        return cache[O]

    bmA = get_bm(A_t)
    bmB = get_bm(B_t)
    bmU = get_bm(U_t)
    raw_ok = raw_boundary_cancellation_holds(sys, A_t, B_t)
    seam = seam_edges(sys, A_t, B_t)

    counts: Counter[tuple[int, int, int]] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_u: dict[tuple[int, int], set[int]] = defaultdict(set)
    for s in rec_states.tolist():
        qa = _component_for_state(sys, bmA, int(s))
        qb = _component_for_state(sys, bmB, int(s))
        qu = _component_for_state(sys, bmU, int(s))
        if qa < 0 or qb < 0 or qu < 0:
            continue
        pair = (qa, qb)
        counts[(qa, qb, qu)] += 1
        pair_counts[pair] += 1
        pair_to_u[pair].add(qu)

    total = sum(pair_counts.values())
    if total == 0:
        gluing_acc = 0.0
        residual = 0.0
    else:
        modal_correct = 0
        residual_acc = 0.0
        for pair, n_pair in pair_counts.items():
            sub = [n for (qa, qb, qu), n in counts.items() if (qa, qb) == pair]
            modal_correct += max(sub)
            residual_acc += n_pair * _entropy_from_counts(sub)
        gluing_acc = float(modal_correct / total)
        residual = float(residual_acc / total)
    ambiguous = sum(1 for us in pair_to_u.values() if len(us) > 1)
    union_classes = {qu for (_qa, _qb, qu) in counts}
    defect = bool(raw_ok and ambiguous > 0 and residual > 1e-9)
    return GluingMeasure(
        A=A_t,
        B=B_t,
        U=U_t,
        raw_cancellation_ok=bool(raw_ok),
        seam_edge_count=int(len(seam)),
        recurrent_state_count=int(len(rec_states)),
        local_pair_count=int(len(pair_counts)),
        union_class_count=int(len(union_classes)),
        ambiguous_pair_count=int(ambiguous),
        gluing_accuracy=float(gluing_acc),
        residual_entropy_bits=float(residual),
        defect_candidate=defect,
    )


def gluing_measure_row(instance: int, ensemble: str, gm: GluingMeasure, k: int, q: int, horizon: int) -> dict:
    return dict(
        instance=int(instance),
        rule_ensemble=str(ensemble),
        k=int(k),
        q=int(q),
        horizon=int(horizon),
        A=str(tuple(gm.A)),
        B=str(tuple(gm.B)),
        U=str(tuple(gm.U)),
        A_size=int(len(gm.A)),
        B_size=int(len(gm.B)),
        U_size=int(len(gm.U)),
        raw_cancellation_ok=bool(gm.raw_cancellation_ok),
        seam_edge_count=int(gm.seam_edge_count),
        recurrent_state_count=int(gm.recurrent_state_count),
        local_pair_count=int(gm.local_pair_count),
        union_class_count=int(gm.union_class_count),
        ambiguous_pair_count=int(gm.ambiguous_pair_count),
        gluing_accuracy=float(gm.gluing_accuracy),
        residual_entropy_bits=float(gm.residual_entropy_bits),
        defect_candidate=bool(gm.defect_candidate),
        A_vertices=json.dumps(list(gm.A)),
        B_vertices=json.dumps(list(gm.B)),
        U_vertices=json.dumps(list(gm.U)),
    )


# --------------------------------------------------------------------------- #
# Sweeps, summaries, plotting
# --------------------------------------------------------------------------- #


def run_relative_geometry_sweep(
    k: int = 6,
    q: int = 2,
    ensembles: Iterable[str] = ("observer_parity_pair", "xor_ring", "random_scc", "constant"),
    n_instances: int = 20,
    observer_min_size: int = 2,
    observer_max_size: int = 3,
    max_observers_per_instance: int | None = 80,
    max_pairs_per_instance: int | None = 120,
    horizon: int = 2,
    max_schedules: int | None = None,
    extra: int | None = None,
    indeg: int = 2,
    base_seed: int = 0,
    require_feedback: bool = True,
    do_shannon: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run observer-boundary and union-gluing tests over closed systems."""
    boundary_rows: list[dict] = []
    gluing_rows: list[dict] = []
    ensembles = tuple(str(e).strip() for e in ensembles if str(e).strip())
    for ens_i, ens in enumerate(ensembles):
        for inst in range(int(n_instances)):
            seed = int(base_seed + 1000003 * ens_i + inst)
            rng = np.random.default_rng(seed)
            sys = make_closed_system(k=k, q=q, rng=rng, ensemble=ens, extra=extra, indeg=indeg)
            schedules = schedule_sample(sys.k, rng, max_schedules=max_schedules)
            step_maps = step_maps_for_schedules(sys, schedules)
            rec = recurrent_states_from_step_maps(step_maps)
            observers = enumerate_observer_candidates(
                sys,
                min_size=observer_min_size,
                max_size=observer_max_size,
                require_feedback=require_feedback,
                max_observers=max_observers_per_instance,
            )
            cache: dict[tuple[int, ...], BoundaryMeasure] = {}
            # Always include a full-system row to certify ∂V=∅.
            full = tuple(range(sys.k))
            bm_full = boundary_measure(sys, full, horizon=horizon, step_maps=step_maps, recurrent_states=rec, schedules=schedules, rng=rng, do_shannon=False)
            boundary_rows.append(boundary_measure_row(inst, ens, bm_full, sys.k, sys.q, horizon, len(schedules)))
            for O in observers:
                bm = boundary_measure(
                    sys, O, horizon=horizon, step_maps=step_maps, recurrent_states=rec,
                    schedules=schedules, rng=rng, do_shannon=do_shannon,
                )
                cache[O] = bm
                boundary_rows.append(boundary_measure_row(inst, ens, bm, sys.k, sys.q, horizon, len(schedules)))
            # Disjoint observer-union tests.  Prefer pairs with actual seams.
            pair_candidates = []
            for A, B in itertools.combinations(observers, 2):
                if set(A) & set(B):
                    continue
                if len(set(A) | set(B)) > sys.k:
                    continue
                pair_candidates.append((A, B, len(seam_edges(sys, A, B))))
            pair_candidates.sort(key=lambda x: (-x[2], len(x[0]) + len(x[1]), x[0], x[1]))
            if max_pairs_per_instance is not None:
                pair_candidates = pair_candidates[: int(max_pairs_per_instance)]
            for A, B, _nseam in pair_candidates:
                gm = gluing_measure(
                    sys, A, B, horizon=horizon, step_maps=step_maps, recurrent_states=rec,
                    schedules=schedules, rng=rng, do_shannon=False, cache=cache,
                )
                gluing_rows.append(gluing_measure_row(inst, ens, gm, sys.k, sys.q, horizon))
            if verbose:
                print(f"relative-geometry ens={ens} inst={inst+1}/{n_instances} observers={len(observers)} pairs={len(pair_candidates)} rec={len(rec)}")
    bdf = pd.DataFrame(boundary_rows)
    gdf = pd.DataFrame(gluing_rows)
    summary = summarize_relative_geometry(bdf, gdf)
    return bdf, gdf, summary


def _safe_mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    return float(series.replace([np.inf, -np.inf], np.nan).mean())


def summarize_relative_geometry(boundary_df: pd.DataFrame, gluing_df: pd.DataFrame | None = None) -> dict:
    if boundary_df is None or boundary_df.empty:
        return {"verdict": "NO OBSERVER-RELATIVE BOUNDARY DATA", "n_boundary_rows": 0, "n_gluing_rows": 0}
    proper = boundary_df[boundary_df["observer_size"] < boundary_df["k"]].copy()
    full = boundary_df[boundary_df["observer_size"] == boundary_df["k"]].copy()
    by_ensemble = []
    for ens, g in proper.groupby("rule_ensemble"):
        by_ensemble.append(dict(
            rule_ensemble=str(ens),
            n_observers=int(len(g)),
            nontrivial_boundary_fraction=float(np.mean(g["boundary_nontrivial"].astype(bool))) if len(g) else float("nan"),
            binary_c2_like_fraction=float(np.mean(g["boundary_binary_c2_like"].astype(bool))) if len(g) else float("nan"),
            mean_boundary_quotient_size=float(g["boundary_quotient_size"].replace([np.inf, -np.inf], np.nan).mean()),
            mean_boundary_quotient_entropy_bits=float(g["boundary_quotient_entropy_bits"].replace([np.inf, -np.inf], np.nan).mean()),
            mean_boundary_mi_bits=float(g["boundary_mi_bits"].replace([np.inf, -np.inf], np.nan).mean()),
            mean_live_site_fraction=float(g["live_site_fraction"].replace([np.inf, -np.inf], np.nan).mean()),
            mean_effective_cycle_rank=float(g["effective_cycle_rank"].replace([np.inf, -np.inf], np.nan).mean()),
        ))
    gluing_summary = []
    defect_fraction = float("nan")
    mean_residual = float("nan")
    raw_cancel_fraction = float("nan")
    if gluing_df is not None and not gluing_df.empty:
        defect_fraction = float(np.mean(gluing_df["defect_candidate"].astype(bool)))
        mean_residual = float(gluing_df["residual_entropy_bits"].replace([np.inf, -np.inf], np.nan).mean())
        raw_cancel_fraction = float(np.mean(gluing_df["raw_cancellation_ok"].astype(bool)))
        for ens, g in gluing_df.groupby("rule_ensemble"):
            gluing_summary.append(dict(
                rule_ensemble=str(ens),
                n_pairs=int(len(g)),
                raw_cancellation_fraction=float(np.mean(g["raw_cancellation_ok"].astype(bool))) if len(g) else float("nan"),
                mean_gluing_accuracy=float(g["gluing_accuracy"].replace([np.inf, -np.inf], np.nan).mean()),
                mean_residual_entropy_bits=float(g["residual_entropy_bits"].replace([np.inf, -np.inf], np.nan).mean()),
                defect_candidate_fraction=float(np.mean(g["defect_candidate"].astype(bool))) if len(g) else float("nan"),
                mean_seam_edge_count=float(g["seam_edge_count"].replace([np.inf, -np.inf], np.nan).mean()),
            ))
    full_empty_ok = bool(np.all(full["raw_boundary_size"].to_numpy(dtype=int) == 0)) if len(full) else False
    nontriv_frac = float(np.mean(proper["boundary_nontrivial"].astype(bool))) if len(proper) else 0.0
    c2_frac = float(np.mean(proper["boundary_binary_c2_like"].astype(bool))) if len(proper) else 0.0
    if not full_empty_ok:
        verdict = "BOUNDARY AXIOM FAILURE: full closed system has nonempty boundary"
    elif len(proper) == 0:
        verdict = "NO PROPER OBSERVER CANDIDATES"
    elif nontriv_frac > 0 and c2_frac >= 0.5 * nontriv_frac:
        verdict = "OBSERVER-RELATIVE C2 BOUNDARY SIGNAL"
    elif nontriv_frac > 0:
        verdict = "OBSERVER-RELATIVE NONTRIVIAL BOUNDARY SIGNAL"
    else:
        verdict = "NO RESOLVED OBSERVER-RELATIVE BOUNDARY SIGNAL"
    return dict(
        verdict=verdict,
        n_boundary_rows=int(len(boundary_df)),
        n_proper_observer_rows=int(len(proper)),
        n_gluing_rows=int(0 if gluing_df is None else len(gluing_df)),
        full_system_boundary_empty=full_empty_ok,
        proper_nontrivial_boundary_fraction=nontriv_frac,
        proper_binary_c2_like_fraction=c2_frac,
        mean_boundary_quotient_size=_safe_mean(proper["boundary_quotient_size"]) if len(proper) else float("nan"),
        mean_boundary_quotient_entropy_bits=_safe_mean(proper["boundary_quotient_entropy_bits"]) if len(proper) else float("nan"),
        mean_boundary_mi_bits=_safe_mean(proper["boundary_mi_bits"]) if len(proper) else float("nan"),
        mean_live_site_fraction=_safe_mean(proper["live_site_fraction"]) if len(proper) else float("nan"),
        raw_cancellation_fraction=raw_cancel_fraction,
        mean_gluing_residual_entropy_bits=mean_residual,
        defect_candidate_fraction=defect_fraction,
        by_ensemble=by_ensemble,
        gluing_by_ensemble=gluing_summary,
    )


def plot_relative_geometry(boundary_df: pd.DataFrame, gluing_df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    proper = boundary_df[boundary_df["observer_size"] < boundary_df["k"]] if not boundary_df.empty else boundary_df
    if proper is not None and not proper.empty:
        g = proper.groupby("rule_ensemble")["boundary_nontrivial"].mean().sort_index()
        ax[0].bar(range(len(g)), g.values)
        ax[0].set_xticks(range(len(g)))
        ax[0].set_xticklabels(g.index, rotation=45, ha="right")
        ax[0].set_ylabel("fraction")
        ax[0].set_title("nontrivial resolved boundaries")
        g2 = proper.groupby("rule_ensemble")["boundary_binary_c2_like"].mean().sort_index()
        ax[1].bar(range(len(g2)), g2.values)
        ax[1].set_xticks(range(len(g2)))
        ax[1].set_xticklabels(g2.index, rotation=45, ha="right")
        ax[1].set_title("binary / C2-like boundaries")
        ax[1].set_ylabel("fraction")
    if gluing_df is not None and not gluing_df.empty:
        g3 = gluing_df.groupby("rule_ensemble")["residual_entropy_bits"].mean().sort_index()
        ax[2].bar(range(len(g3)), g3.values)
        ax[2].set_xticks(range(len(g3)))
        ax[2].set_xticklabels(g3.index, rotation=45, ha="right")
        ax[2].set_title("gluing residual H(Q_U|Q_A,Q_B)")
        ax[2].set_ylabel("bits")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_ensembles(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Observer-relative boundary geometry and gluing audit for closed finite systems.")
    p.add_argument("--mode", choices=["sweep"], default="sweep")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--q", type=int, default=2)
    p.add_argument("--ensembles", default="observer_parity_pair,xor_ring,random_scc,constant")
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--observer-min-size", type=int, default=2)
    p.add_argument("--observer-max-size", type=int, default=3)
    p.add_argument("--max-observers", type=int, default=80)
    p.add_argument("--max-pairs", type=int, default=120)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--max-schedules", type=int, default=0, help="0 means exhaustive all k! schedules")
    p.add_argument("--extra", type=int, default=None)
    p.add_argument("--indeg", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-feedback-required", action="store_true")
    p.add_argument("--no-shannon", action="store_true")
    p.add_argument("--out", default="example_results/relative_geometry_sweep.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    max_schedules = None if int(args.max_schedules) <= 0 else int(args.max_schedules)
    bdf, gdf, summary = run_relative_geometry_sweep(
        k=int(args.k),
        q=int(args.q),
        ensembles=_parse_ensembles(args.ensembles),
        n_instances=int(args.instances),
        observer_min_size=int(args.observer_min_size),
        observer_max_size=int(args.observer_max_size),
        max_observers_per_instance=None if int(args.max_observers) <= 0 else int(args.max_observers),
        max_pairs_per_instance=None if int(args.max_pairs) <= 0 else int(args.max_pairs),
        horizon=int(args.horizon),
        max_schedules=max_schedules,
        extra=args.extra,
        indeg=int(args.indeg),
        base_seed=int(args.seed),
        require_feedback=not bool(args.no_feedback_required),
        do_shannon=not bool(args.no_shannon),
        verbose=not bool(args.quiet),
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        base = os.path.splitext(args.out)[0]
        bdf.to_csv(args.out, index=False)
        gdf.to_csv(base + "_gluing.csv", index=False)
        with open(base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_relative_geometry(bdf, gdf, args.plot)
    print(json.dumps(summary, indent=2))
    if args.out:
        print(f"wrote {args.out}")
        print(f"wrote {os.path.splitext(args.out)[0]}_gluing.csv")
        print(f"wrote {os.path.splitext(args.out)[0]}_summary.json")
    if args.plot:
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
