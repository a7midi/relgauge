"""
observerframebundle.py

Observer-frame bundle audit for closed finite relational systems.

This module is the bridge between observer-relative boundary geometry and
observer-relative holonomy.  The earlier boundary-geometry layer discovers
proper SCC observers and live cut/cycle geometry.  The holonomy layer asks for
transported edge quotients directly.  This module inserts the missing geometric
object:

    observer boundary geometry -> observer frame -> connection -> holonomy.

The observer frame F_O is not supplied as a gauge group.  In the default
``local_charts`` mode, each incident boundary port carries its own endpoint
chart, canonicalized independently on the two sides of an inter-observer
channel.  Edge transports are oriented transition maps between these local
charts.  This is deliberately different from edge-common-factor coordinates:
common-factor coordinates are excellent for agreement/equalizer audits, but
would gauge-flatten a flip edge by naming both endpoints with the same shared
component label.  ``common_factor`` mode remains available as a backward-
compatible flat/equalizer audit.

Path comparability and loop automorphism validity are audited post-hoc on
discovered two-path observer diamonds.  Flat path agreement is reported
separately as a diagnostic.

Important scope
---------------
This is an audit/certifier, not a selection experiment.  It does not reward or
insert C2, C3, S_n, nontrivial holonomy, flux, matter, or plaquettes.  It can be
used by blindobserverconnection.py as a pure consistency objective by selecting
only for live frames, deterministic transported edge maps, branch completion,
and well-defined loop automorphisms.  It does not select for identity holonomy.

CLI examples
------------
Positive controls:

python -m relgauge.observerframebundle 4 --graphs 4 --graph-mode frame_flat_c2 \
  --out example_results/observer_frame_flat.csv

python -m relgauge.observerframebundle 4 --graphs 4 --graph-mode frame_twist_c2 \
  --out example_results/observer_frame_twist.csv

In the default local-charts mode, the twist control is a positive non-flat
control: one branch carries a binary flip, so the loop closes as a nontrivial
C2 automorphism rather than being flattened by shared edge coordinates.  Use
``--frame-coordinate-mode common_factor`` to reproduce the older
agreement-coordinate audit, where this same twist is absorbed as a local
common-factor gauge choice.

Passive random/componented audit:

python -m relgauge.observerframebundle 4 --graphs 50 --vertices 18 \
  --graph-mode componented --components 5 --inter-prob 0.35 \
  --extra-intra-prob 0.25 --out example_results/observer_frame_componented.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # optional at import time; required for CLI CSV output
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import observerboundarygeometry as OBG
    from . import observerrelativeholonomy as ORH
except Exception:  # pragma: no cover
    import observerboundarygeometry as OBG  # type: ignore
    import observerrelativeholonomy as ORH  # type: ignore

Edge = Tuple[int, int]
CompEdge = Tuple[int, int]
PortKey = Tuple[CompEdge, str]  # ((source_comp,target_comp), "source"|"target")


# ---------------------------------------------------------------------------
# Small generic utilities
# ---------------------------------------------------------------------------
def entropy_from_counts(counts: Iterable[int | float]) -> float:
    vals = np.asarray([float(c) for c in counts if float(c) > 0.0], dtype=float)
    if vals.size == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-(p * np.log2(p)).sum())


def encode_tuple(vals: Sequence[int], q: int) -> int:
    code = 0
    for v in vals:
        code = code * q + int(v)
    return int(code)


def all_or_sample_states(q: int, k: int, max_states: int, rng: np.random.Generator) -> List[Tuple[int, ...]]:
    total = q ** k
    if total <= max_states:
        return [tuple(int(x) for x in s) for s in itertools.product(range(q), repeat=k)]
    return [tuple(int(x) for x in rng.integers(0, q, size=k)) for _ in range(max_states)]


def parallel_step(sys: OBG.FiniteRelationalSystem, state: Sequence[int]) -> Tuple[int, ...]:
    return sys.step_parallel(state)


def warm_state_samples(
    sys: OBG.FiniteRelationalSystem,
    rng: np.random.Generator,
    max_states: int = 4096,
    warmup: int = 3,
) -> List[Tuple[int, ...]]:
    """Sample dynamically plausible global states by warming under parallel update.

    For tiny systems this enumerates all states.  For larger systems this samples
    random states.  The warmup is intentionally not treated as physical time; it
    is a finite way to restrict local frame comparisons to states compatible
    with the closed relational update rules rather than arbitrary local
    assignments that may never co-occur.
    """
    states = all_or_sample_states(sys.q, sys.k, max_states, rng)
    out: Set[Tuple[int, ...]] = set()
    for s in states:
        cur = tuple(int(x) for x in s)
        for _ in range(max(0, int(warmup))):
            cur = parallel_step(sys, cur)
        out.add(cur)
    return sorted(out)


def best_deterministic_map_from_pairs(
    pairs: Iterable[Tuple[int, int]],
    n_src: int,
    n_tgt: int,
) -> Tuple[Dict[int, int], float, bool, float, float]:
    """Best deterministic map from observed pairs.

    Returns mapping, accuracy, bijective?, source entropy, coverage fraction.
    """
    if n_src <= 0 or n_tgt <= 0:
        return {}, 0.0, False, 0.0, 0.0
    counts = np.zeros((int(n_src), int(n_tgt)), dtype=np.int64)
    for a, b in pairs:
        if 0 <= int(a) < n_src and 0 <= int(b) < n_tgt:
            counts[int(a), int(b)] += 1
    total = int(counts.sum())
    if total == 0:
        return {}, 0.0, False, 0.0, 0.0
    mapping: Dict[int, int] = {}
    correct = 0
    covered_rows = 0
    for a in range(n_src):
        row = counts[a]
        if int(row.sum()) > 0:
            covered_rows += 1
            b = int(row.argmax())
            mapping[a] = b
            correct += int(row[b])
    acc = float(correct / max(1, total))
    coverage = float(covered_rows / max(1, n_src))
    bij = bool(n_src == n_tgt and covered_rows == n_src and len(set(mapping.values())) == n_tgt)
    h_src = entropy_from_counts(counts.sum(axis=1))
    return mapping, acc, bij, h_src, coverage


def compose_maps(f: Dict[int, int], g: Dict[int, int]) -> Dict[int, int]:
    """g after f: x -> g[f[x]]."""
    out: Dict[int, int] = {}
    for a, b in f.items():
        if b in g:
            out[int(a)] = int(g[b])
    return out


def invert_permutation(m: Dict[int, int], n: int) -> Optional[Dict[int, int]]:
    if set(m.keys()) != set(range(n)) or set(m.values()) != set(range(n)):
        return None
    return {int(v): int(k) for k, v in m.items()}


def permutation_cycle_type(perm: Dict[int, int], n: int) -> str:
    if n <= 0 or set(perm.keys()) != set(range(n)) or set(perm.values()) != set(range(n)):
        return "nonbijective"
    seen: Set[int] = set()
    lens: List[int] = []
    for i in range(n):
        if i in seen:
            continue
        cur = i
        L = 0
        while cur not in seen:
            seen.add(cur)
            L += 1
            cur = int(perm[cur])
        lens.append(L)
    lens = sorted(lens, reverse=True)
    if all(x == 1 for x in lens):
        return "identity"
    return "cycle_" + "_".join(str(x) for x in lens)


def generated_group_order(generators: List[Dict[int, int]], n: int, cap: int = 200000) -> Tuple[int, str]:
    gens: List[Tuple[int, ...]] = []
    for g in generators:
        if set(g.keys()) == set(range(n)) and set(g.values()) == set(range(n)):
            gens.append(tuple(int(g[i]) for i in range(n)))
    if not gens:
        return 0, "none"
    ident = tuple(range(n))
    seen = {ident}
    dq = deque([ident])

    def comp(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(int(b[a[i]]) for i in range(n))

    while dq and len(seen) < cap:
        cur = dq.popleft()
        for g in gens:
            for h in (comp(cur, g), comp(g, cur)):
                if h not in seen:
                    seen.add(h)
                    dq.append(h)
    order = len(seen)
    if order == 1:
        name = "trivial"
    elif order == 2:
        name = "C2"
    elif order == 3:
        name = "C3"
    elif order == 4:
        name = "C4_or_V4"
    elif order == 6 and n == 3:
        name = "S3_or_D3"
    elif order == math.factorial(n):
        name = f"S{n}"
    else:
        name = f"order_{order}"
    return int(order), name


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Positive-control observer-frame diamond systems
# ---------------------------------------------------------------------------
def _copy_table(q: int, arity: int, copy_arg: int = 0, post=None) -> np.ndarray:
    """Table that copies one predecessor using OBG's big-endian code order."""
    arr = np.empty(q ** arity, dtype=np.int16)
    for code in range(q ** arity):
        vals = []
        x = int(code)
        # OBG.eval_vertex_from_assignment builds code by repeated
        # code = code*q + pred_value, so pred[0] is most significant.
        for j in range(arity):
            power = q ** (arity - 1 - j)
            vals.append((x // power) % q)
        v = vals[int(copy_arg)] if vals else 0
        if post is not None:
            v = post(v) % q
        arr[code] = int(v)
    return arr


def _twist_table(q: int, arity: int, copy_arg: int = 0) -> np.ndarray:
    # A binary parity twist inside q symbols: copy parity then flip it.
    # Keeping the output binary is important for the q>2 positive control:
    # otherwise a local all-input channel audit would see a four-class flip
    # even though the dynamically transported frame is binary.
    return _copy_table(q, arity, copy_arg=copy_arg, post=lambda x: (x % 2) ^ 1)


def make_frame_diamond_control(q: int = 4, twist: bool = False) -> OBG.FiniteRelationalSystem:
    """Closed four-observer diamond positive control.

    Components are A,B,C,D, each with two vertices and an internal 2-cycle so
    every component is a genuine SCC observer.  Inter-component condensation is
    A->B, A->C, B->D, C->D.  The transported label is the parity of A's boundary
    source; q may be larger than 2.  If twist=True, the C->D branch flips the
    binary label.  The maximal observer frame may absorb this as a local gauge
    choice; this is useful as a gauge-flattening sanity check.
    """
    if q < 2:
        raise ValueError("q must be >=2")
    k = 8
    # Vertex layout: A0,A1,B0,B1,C0,C1,D0,D1
    pred_sets: List[List[int]] = [[] for _ in range(k)]
    # Internal feedback cycles for SCCs.
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        pred_sets[a].append(b)
        pred_sets[b].append(a)
    # Inter-observer micro-edges.
    # A0 -> B0, A0 -> C0, B0 -> D0, C0 -> D1.
    pred_sets[2].append(0)
    pred_sets[4].append(0)
    pred_sets[6].append(2)
    pred_sets[7].append(4)
    preds = [tuple(ps) for ps in pred_sets]

    tables: List[np.ndarray] = []
    for v, ps in enumerate(preds):
        if v == 0:
            # A0 copies A1: source memory is free/stable under 2-cycle.
            tables.append(_copy_table(q, len(ps), copy_arg=0))
        elif v == 1:
            tables.append(_copy_table(q, len(ps), copy_arg=0))
        elif v == 2:
            # B0 depends on (B1,A0); copy parity of A0 (arg 1).
            tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        elif v == 3:
            tables.append(_copy_table(q, len(ps), copy_arg=0))
        elif v == 4:
            # C0 depends on (C1,A0); copy parity of A0.
            tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        elif v == 5:
            tables.append(_copy_table(q, len(ps), copy_arg=0))
        elif v == 6:
            # D0 depends on (D1,B0); copy B0 parity.
            tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        elif v == 7:
            # D1 depends on (D0,C0); copy C0 parity, optional flip.
            if twist:
                tables.append(_twist_table(q, len(ps), copy_arg=1))
            else:
                tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
    return OBG.FiniteRelationalSystem(q=q, preds=preds, tables=tables)


def make_random_frame_diamond(
    q: int,
    rng: np.random.Generator,
    max_pred: Optional[int] = None,
) -> OBG.FiniteRelationalSystem:
    """Same four-observer diamond topology as the controls, but random rules."""
    base = make_frame_diamond_control(q=q, twist=False)
    tables = [OBG._random_table(q, len(ps), rng) for ps in base.preds]
    return OBG.FiniteRelationalSystem(q=q, preds=list(base.preds), tables=tables)


def _add_internal_cycles(pred_sets: List[List[int]], comps: Sequence[Sequence[int]]) -> None:
    """Give every component an internal directed cycle in the graph."""
    for comp in comps:
        c = list(comp)
        if len(c) <= 1:
            continue
        for a, b in zip(c, c[1:] + c[:1]):
            pred_sets[int(b)].append(int(a))


def _component_first_vertices(comps: Sequence[Sequence[int]]) -> List[int]:
    return [int(list(c)[0]) for c in comps]


def make_frame_theta_control(
    q: int = 4,
    twists: Sequence[bool] = (False, False, True),
) -> OBG.FiniteRelationalSystem:
    """Closed five-observer theta graph positive control.

    Observer condensation graph:

        A -> B -> D
        A -> C -> D
        A -> E -> D

    There are three branches from A to D and therefore two independent cycles.
    Each observer is a genuine SCC.  The three branch-to-sink transports carry
    a binary parity label; entries in ``twists`` flip the corresponding branch
    at the sink.  With one flipped branch, two of the three pairwise loop
    holonomies are nontrivial C2, as required by the theta relation.
    """
    if q < 2:
        raise ValueError("q must be >=2")
    twists = tuple(bool(x) for x in list(twists)[:3] + [False] * max(0, 3 - len(twists)))
    # A(2), B(2), C(2), E(2), D(3) = 11 vertices.
    comps = [range(0, 2), range(2, 4), range(4, 6), range(6, 8), range(8, 11)]
    k = 11
    pred_sets: List[List[int]] = [[] for _ in range(k)]
    _add_internal_cycles(pred_sets, comps)
    # A0 -> branch0/1/2 first vertices.
    pred_sets[2].append(0)
    pred_sets[4].append(0)
    pred_sets[6].append(0)
    # Branch first vertices -> three distinct sink vertices D0,D1,D2.
    pred_sets[8].append(2)
    pred_sets[9].append(4)
    pred_sets[10].append(6)
    preds = [tuple(ps) for ps in pred_sets]

    tables: List[np.ndarray] = []
    for v, ps in enumerate(preds):
        if v in (0, 1):
            tables.append(_copy_table(q, len(ps), copy_arg=0))
        elif v in (2, 4, 6):
            # Branch input vertices depend on (internal predecessor, A0).
            tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        elif v in (3, 5, 7):
            tables.append(_copy_table(q, len(ps), copy_arg=0))
        elif v in (8, 9, 10):
            branch_index = {8: 0, 9: 1, 10: 2}[v]
            if twists[branch_index]:
                tables.append(_twist_table(q, len(ps), copy_arg=1))
            else:
                tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        else:  # pragma: no cover
            tables.append(_copy_table(q, len(ps), copy_arg=0))
    return OBG.FiniteRelationalSystem(q=q, preds=preds, tables=tables)


def make_random_frame_theta(q: int, rng: np.random.Generator, max_pred: Optional[int] = None) -> OBG.FiniteRelationalSystem:
    """Theta observer graph with random local rule tables."""
    base = make_frame_theta_control(q=q, twists=(False, False, False))
    tables = [OBG._random_table(q, len(ps), rng) for ps in base.preds]
    return OBG.FiniteRelationalSystem(q=q, preds=list(base.preds), tables=tables)


def make_frame_double_diamond_control(
    q: int = 4,
    twists: Sequence[bool] = (True, False),
) -> OBG.FiniteRelationalSystem:
    """Two coupled observer diamonds in series.

    Condensation graph:

        A -> B -> D -> E -> G
        A -> C -> D -> F -> G

    The two diamonds share the middle observer D.  ``twists[0]`` flips the lower
    branch of the first diamond (C->D), and ``twists[1]`` flips the lower branch
    of the second diamond (F->G).  This is the smallest controlled arena where
    section-obstruction representatives can potentially move between two loops.
    """
    if q < 2:
        raise ValueError("q must be >=2")
    tw0 = bool(twists[0]) if len(twists) > 0 else False
    tw1 = bool(twists[1]) if len(twists) > 1 else False
    # Seven two-vertex observers A,B,C,D,E,F,G.
    comps = [range(2*i, 2*i+2) for i in range(7)]
    k = 14
    pred_sets: List[List[int]] = [[] for _ in range(k)]
    _add_internal_cycles(pred_sets, comps)
    # A->B,C; B,C->D; D->E,F; E,F->G.
    pred_sets[2].append(0)    # A0 -> B0
    pred_sets[4].append(0)    # A0 -> C0
    pred_sets[6].append(2)    # B0 -> D0
    pred_sets[7].append(4)    # C0 -> D1
    pred_sets[8].append(6)    # D0 -> E0
    pred_sets[10].append(7)   # D1 -> F0
    pred_sets[12].append(8)   # E0 -> G0
    pred_sets[13].append(10)  # F0 -> G1
    preds = [tuple(ps) for ps in pred_sets]

    tables: List[np.ndarray] = []
    for v, ps in enumerate(preds):
        if v in (0, 1):
            tables.append(_copy_table(q, len(ps), copy_arg=0))
        elif v in (2, 4, 8, 10):
            tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        elif v in (3, 5, 9, 11):
            tables.append(_copy_table(q, len(ps), copy_arg=0))
        elif v == 6:
            tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        elif v == 7:
            tables.append(_twist_table(q, len(ps), copy_arg=1) if tw0 else _copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        elif v == 12:
            tables.append(_copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        elif v == 13:
            tables.append(_twist_table(q, len(ps), copy_arg=1) if tw1 else _copy_table(q, len(ps), copy_arg=1, post=lambda x: x % 2))
        else:  # pragma: no cover
            tables.append(_copy_table(q, len(ps), copy_arg=0))
    return OBG.FiniteRelationalSystem(q=q, preds=preds, tables=tables)


def make_random_frame_double_diamond(q: int, rng: np.random.Generator, max_pred: Optional[int] = None) -> OBG.FiniteRelationalSystem:
    """Two-coupled-diamond observer graph with random local rule tables."""
    base = make_frame_double_diamond_control(q=q, twists=(False, False))
    tables = [OBG._random_table(q, len(ps), rng) for ps in base.preds]
    return OBG.FiniteRelationalSystem(q=q, preds=list(base.preds), tables=tables)


# ---------------------------------------------------------------------------
# Frame/connection dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ObserverFrame:
    observer_id: int
    vertices: Tuple[int, ...]
    ports: Tuple[PortKey, ...]
    n_classes: int
    entropy_bits: float
    live: bool
    label_counts: Dict[int, int]
    port_label_to_frame: Dict[Tuple[PortKey, int], int]
    port_label_counts: Dict[Tuple[PortKey, int], int]
    coverage_fraction: float


@dataclass
class FrameTransport:
    comp_edge: CompEdge
    source_frame: int
    target_frame: int
    n_source_classes: int
    n_target_classes: int
    n_source_ports: int
    n_target_ports: int
    mapping: Dict[int, int]
    accuracy: float
    coverage_fraction: float
    bijective: bool
    source_entropy_bits: float
    live: bool
    true_live: bool


@dataclass
class EdgeChart:
    """Endpoint-local chart for one inter-observer boundary channel.

    ``common_factor`` mode uses the old shared equalizer coordinates: source
    and target labels are the same connected-component labels, so the edge map
    is identity.

    ``local_charts`` mode uses the same exact channel factor to decide which
    endpoint codes belong to a live finite quotient, but it canonicalizes the
    source endpoint and target endpoint independently by their own local code
    order.  The oriented edge map is then the transition function between those
    two local charts.  For a perfect flip relation, this produces a flip map
    instead of silently renaming the target endpoint to make the map identity.
    """

    comp_edge: CompEdge
    source_nodes: Tuple[int, ...]
    target_nodes: Tuple[int, ...]
    n_classes: int
    source_label: Dict[int, int]
    target_label: Dict[int, int]
    edge_map: Dict[int, int]
    live: bool
    mode: str



# ---------------------------------------------------------------------------
# SCC/edge extraction helpers
# ---------------------------------------------------------------------------
def component_map(comps: List[List[int]]) -> Dict[int, int]:
    return {int(v): int(ci) for ci, comp in enumerate(comps) for v in comp}


def directed_pair_edges(sys: OBG.FiniteRelationalSystem, comp_of: Dict[int, int]) -> Dict[CompEdge, List[Edge]]:
    out: Dict[CompEdge, List[Edge]] = defaultdict(list)
    for u, v in sys.edges():
        cu, cv = comp_of[int(u)], comp_of[int(v)]
        if cu != cv:
            out[(cu, cv)].append((int(u), int(v)))
    return dict(out)


def port_side_label(eq: EdgeChart | ORH.EdgeQuotient, side: str, state: Sequence[int], q: int) -> Optional[int]:
    if side == "source":
        nodes = eq.source_nodes
        lab = eq.source_label
    elif side == "target":
        nodes = eq.target_nodes
        lab = eq.target_label
    else:
        raise ValueError(side)
    code = encode_tuple([int(state[n]) for n in nodes], q)
    val = lab.get(code)
    return None if val is None else int(val)


def _side_canonical_labels(side_label: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Relabel one endpoint's common-factor classes by local code order.

    ``side_label`` maps endpoint raw codes to shared common-factor component
    ids.  We keep the induced partition but give the endpoint its own local
    coordinate names by sorting classes according to the smallest local raw code
    that realizes them.  This is a gauge choice for the local chart, not a
    physical target.
    """
    groups: Dict[int, List[int]] = defaultdict(list)
    for code, cls in side_label.items():
        groups[int(cls)].append(int(code))
    ordered = sorted(groups, key=lambda c: (min(groups[c]), c))
    comp_to_local = {int(c): int(i) for i, c in enumerate(ordered)}
    local_label = {int(code): int(comp_to_local[int(cls)]) for code, cls in side_label.items()}
    return local_label, comp_to_local


def edge_chart_from_common_factor(eq: ORH.EdgeQuotient, mode: str = "local_charts") -> EdgeChart:
    """Convert an exact edge common factor into endpoint chart coordinates."""
    mode = str(mode).lower()
    if mode not in {"common_factor", "local_charts"}:
        raise ValueError(f"unknown frame coordinate mode: {mode}")
    if eq.n_classes <= 0:
        return EdgeChart(eq.comp_edge, eq.source_nodes, eq.target_nodes, 0, {}, {}, {}, False, mode)

    if mode == "common_factor":
        n = int(eq.n_classes)
        return EdgeChart(
            comp_edge=eq.comp_edge,
            source_nodes=eq.source_nodes,
            target_nodes=eq.target_nodes,
            n_classes=n,
            source_label={int(k): int(v) for k, v in eq.source_label.items()},
            target_label={int(k): int(v) for k, v in eq.target_label.items()},
            edge_map={i: i for i in range(n)},
            live=bool(eq.live),
            mode=mode,
        )

    src_label, src_comp_to_local = _side_canonical_labels(eq.source_label)
    tgt_label, tgt_comp_to_local = _side_canonical_labels(eq.target_label)
    common_components = sorted(set(src_comp_to_local) & set(tgt_comp_to_local))
    edge_map: Dict[int, int] = {}
    for c in common_components:
        edge_map[int(src_comp_to_local[c])] = int(tgt_comp_to_local[c])
    n = int(max(len(src_comp_to_local), len(tgt_comp_to_local), eq.n_classes))
    # The chart is live only when the oriented endpoint map is a bijection on
    # the induced class alphabet.  This keeps frame transport exact while still
    # exposing flips/twists as non-identity maps.
    bij = bool(set(edge_map.keys()) == set(range(n)) and set(edge_map.values()) == set(range(n)))
    return EdgeChart(
        comp_edge=eq.comp_edge,
        source_nodes=eq.source_nodes,
        target_nodes=eq.target_nodes,
        n_classes=n,
        source_label=src_label,
        target_label=tgt_label,
        edge_map=edge_map,
        live=bool(eq.live and bij),
        mode=mode,
    )


def build_edge_charts(edge_q: Dict[CompEdge, ORH.EdgeQuotient], mode: str = "local_charts") -> Dict[CompEdge, EdgeChart]:
    return {ce: edge_chart_from_common_factor(eq, mode=mode) for ce, eq in edge_q.items()}


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------
def build_observer_frames_common_factor(
    sys: OBG.FiniteRelationalSystem,
    comps: List[List[int]],
    edge_q: Dict[CompEdge, EdgeChart],
    state_samples: Sequence[Sequence[int]],
    min_frame_classes: int = 2,
    min_frame_entropy: float = 0.25,
    require_live_edge: bool = True,
) -> Dict[int, ObserverFrame]:
    """Extract observer frames as common factors of incident port labels."""
    incident: Dict[int, List[PortKey]] = defaultdict(list)
    for ce, eq in edge_q.items():
        if require_live_edge and not eq.live:
            continue
        a, b = ce
        incident[a].append((ce, "source"))
        incident[b].append((ce, "target"))

    frames: Dict[int, ObserverFrame] = {}
    for oid, comp in enumerate(comps):
        ports = tuple(sorted(incident.get(oid, []), key=lambda x: (x[0], x[1])))
        if not ports:
            frames[oid] = ObserverFrame(
                observer_id=oid,
                vertices=tuple(comp),
                ports=tuple(),
                n_classes=0,
                entropy_bits=0.0,
                live=False,
                label_counts={},
                port_label_to_frame={},
                port_label_counts={},
                coverage_fraction=0.0,
            )
            continue

        # Node in multipartite co-occurrence graph: (port_index, edge-label).
        node_index: Dict[Tuple[int, int], int] = {}
        occurrences: List[List[Tuple[int, int]]] = []
        port_label_counts: Counter = Counter()
        for s in state_samples:
            labels: List[Tuple[int, int]] = []
            for pi, port in enumerate(ports):
                ce, side = port
                eq = edge_q[ce]
                lab = port_side_label(eq, side, s, sys.q)
                if lab is None:
                    continue
                key = (pi, int(lab))
                if key not in node_index:
                    node_index[key] = len(node_index)
                labels.append(key)
                port_label_counts[(port, int(lab))] += 1
            if labels:
                occurrences.append(labels)

        if not node_index:
            frames[oid] = ObserverFrame(oid, tuple(comp), ports, 0, 0.0, False, {}, {}, {}, 0.0)
            continue
        uf = UnionFind(len(node_index))
        for labels in occurrences:
            first = node_index[labels[0]]
            for lab in labels[1:]:
                uf.union(first, node_index[lab])

        roots = sorted(set(uf.find(i) for i in range(len(node_index))))
        root_to_frame = {r: j for j, r in enumerate(roots)}
        port_label_to_frame: Dict[Tuple[PortKey, int], int] = {}
        for (pi, lab), idx in node_index.items():
            port_label_to_frame[(ports[pi], int(lab))] = int(root_to_frame[uf.find(idx)])

        # Count frame label usage across sampled local occurrences.
        frame_counts: Counter = Counter()
        total_occ = 0
        for labels in occurrences:
            used = sorted({port_label_to_frame[(ports[pi], int(lab))] for pi, lab in labels})
            for fc in used:
                frame_counts[int(fc)] += 1
                total_occ += 1
        h = entropy_from_counts(frame_counts.values())
        n_classes = len(frame_counts)
        coverage = float(total_occ / max(1, len(state_samples) * max(1, len(ports))))
        live = bool(n_classes >= min_frame_classes and h >= min_frame_entropy and coverage > 0.0)
        frames[oid] = ObserverFrame(
            observer_id=oid,
            vertices=tuple(int(v) for v in comp),
            ports=ports,
            n_classes=int(n_classes),
            entropy_bits=float(h),
            live=live,
            label_counts={int(k): int(v) for k, v in frame_counts.items()},
            port_label_to_frame=port_label_to_frame,
            port_label_counts={k: int(v) for k, v in port_label_counts.items()},
            coverage_fraction=coverage,
        )
    return frames


def build_observer_frames_local_charts(
    sys: OBG.FiniteRelationalSystem,
    comps: List[List[int]],
    edge_q: Dict[CompEdge, EdgeChart],
    state_samples: Sequence[Sequence[int]],
    min_frame_classes: int = 2,
    min_frame_entropy: float = 0.25,
    require_live_edge: bool = True,
) -> Dict[int, ObserverFrame]:
    """Build observer frames as endpoint-local charts, not edge equalizers.

    Each live incident port supplies a finite endpoint alphabet.  The observer's
    local frame is the common coordinate alphabet size shared by the largest
    set of incident ports.  The per-port chart is canonical identity in the
    endpoint's independently canonicalized local labels.  This intentionally
    avoids merging co-occurring labels across different ports, because that
    co-occurrence/common-factor merge would absorb a one-branch flip at the
    sink and flatten the twist before holonomy is computed.
    """
    incident: Dict[int, List[PortKey]] = defaultdict(list)
    for ce, eq in edge_q.items():
        if require_live_edge and not eq.live:
            continue
        a, b = ce
        incident[int(a)].append((ce, "source"))
        incident[int(b)].append((ce, "target"))

    frames: Dict[int, ObserverFrame] = {}
    for oid, comp in enumerate(comps):
        raw_ports = tuple(sorted(incident.get(oid, []), key=lambda x: (x[0], x[1])))
        if not raw_ports:
            frames[oid] = ObserverFrame(oid, tuple(comp), tuple(), 0, 0.0, False, {}, {}, {}, 0.0)
            continue

        # Use the most common endpoint chart size among incident live ports.
        by_n: Dict[int, List[PortKey]] = defaultdict(list)
        for port in raw_ports:
            ce, _side = port
            n = int(edge_q[ce].n_classes)
            if n > 0:
                by_n[n].append(port)
        if not by_n:
            frames[oid] = ObserverFrame(oid, tuple(comp), tuple(), 0, 0.0, False, {}, {}, {}, 0.0)
            continue
        n_classes = sorted(by_n, key=lambda n: (-len(by_n[n]), n))[0]
        ports = tuple(sorted(by_n[n_classes], key=lambda x: (x[0], x[1])))

        port_label_to_frame: Dict[Tuple[PortKey, int], int] = {}
        for port in ports:
            for lab in range(int(n_classes)):
                port_label_to_frame[(port, int(lab))] = int(lab)

        frame_counts: Counter = Counter()
        port_label_counts: Counter = Counter()
        total_seen = 0
        total_possible = max(1, len(state_samples) * max(1, len(ports)))
        for s in state_samples:
            for port in ports:
                ce, side = port
                lab = port_side_label(edge_q[ce], side, s, sys.q)
                if lab is None or int(lab) not in range(int(n_classes)):
                    continue
                lab = int(lab)
                frame_counts[lab] += 1
                port_label_counts[(port, lab)] += 1
                total_seen += 1

        h = entropy_from_counts(frame_counts.values())
        coverage = float(total_seen / total_possible)
        used_classes = len(frame_counts)
        live = bool(n_classes >= min_frame_classes and used_classes >= min_frame_classes and h >= min_frame_entropy and coverage > 0.0)
        frames[oid] = ObserverFrame(
            observer_id=oid,
            vertices=tuple(int(v) for v in comp),
            ports=ports,
            n_classes=int(n_classes),
            entropy_bits=float(h),
            live=live,
            label_counts={int(k): int(v) for k, v in frame_counts.items()},
            port_label_to_frame=port_label_to_frame,
            port_label_counts={k: int(v) for k, v in port_label_counts.items()},
            coverage_fraction=coverage,
        )
    return frames


def build_observer_frames(
    sys: OBG.FiniteRelationalSystem,
    comps: List[List[int]],
    edge_q: Dict[CompEdge, EdgeChart],
    state_samples: Sequence[Sequence[int]],
    min_frame_classes: int = 2,
    min_frame_entropy: float = 0.25,
    require_live_edge: bool = True,
    frame_coordinate_mode: str = "local_charts",
) -> Dict[int, ObserverFrame]:
    mode = str(frame_coordinate_mode).lower()
    if mode == "common_factor":
        return build_observer_frames_common_factor(
            sys, comps, edge_q, state_samples,
            min_frame_classes=min_frame_classes,
            min_frame_entropy=min_frame_entropy,
            require_live_edge=require_live_edge,
        )
    if mode == "local_charts":
        return build_observer_frames_local_charts(
            sys, comps, edge_q, state_samples,
            min_frame_classes=min_frame_classes,
            min_frame_entropy=min_frame_entropy,
            require_live_edge=require_live_edge,
        )
    raise ValueError(f"unknown frame coordinate mode: {frame_coordinate_mode}")


def build_frame_transports(
    frames: Dict[int, ObserverFrame],
    edge_q: Dict[CompEdge, EdgeChart],
    min_accuracy: float = 0.999,
    min_coverage: float = 0.999,
    min_frame_ports: int = 2,
) -> Dict[CompEdge, FrameTransport]:
    """Induce maps between observer frames along inter-observer channels.

    ``live`` means the edge label maps deterministically between whatever
    frame-like labels exist at the two endpoints.  ``true_live`` is stricter:
    the two endpoint frames must each coordinate at least ``min_frame_ports``
    incident boundary ports.  This separates a one-port edge label from a real
    local observer frame.
    """
    transports: Dict[CompEdge, FrameTransport] = {}

    def empty_transport(ce: CompEdge, fa: Optional[ObserverFrame], fb: Optional[ObserverFrame]) -> FrameTransport:
        return FrameTransport(
            comp_edge=ce,
            source_frame=ce[0],
            target_frame=ce[1],
            n_source_classes=fa.n_classes if fa else 0,
            n_target_classes=fb.n_classes if fb else 0,
            n_source_ports=len(fa.ports) if fa else 0,
            n_target_ports=len(fb.ports) if fb else 0,
            mapping={},
            accuracy=0.0,
            coverage_fraction=0.0,
            bijective=False,
            source_entropy_bits=0.0,
            live=False,
            true_live=False,
        )

    for ce, eq in edge_q.items():
        a, b = ce
        fa = frames.get(a)
        fb = frames.get(b)
        if fa is None or fb is None or not fa.live or not fb.live or eq.n_classes <= 0:
            transports[ce] = empty_transport(ce, fa, fb)
            continue
        p_src: PortKey = (ce, "source")
        p_tgt: PortKey = (ce, "target")
        pairs: List[Tuple[int, int]] = []
        # In common-factor mode this edge map is identity.  In local-charts
        # mode it is an oriented transition between independently named
        # endpoint charts, so a flip remains a flip.
        for src_lab, tgt_lab in sorted(eq.edge_map.items()):
            ka = (p_src, int(src_lab))
            kb = (p_tgt, int(tgt_lab))
            if ka in fa.port_label_to_frame and kb in fb.port_label_to_frame:
                ca = int(fa.port_label_to_frame[ka])
                cb = int(fb.port_label_to_frame[kb])
                w = max(1, min(fa.port_label_counts.get(ka, 1), fb.port_label_counts.get(kb, 1)))
                for _ in range(int(w)):
                    pairs.append((ca, cb))
        mapping, acc, bij, h_src, coverage = best_deterministic_map_from_pairs(pairs, fa.n_classes, fb.n_classes)
        live = bool(acc >= min_accuracy and coverage >= min_coverage and len(mapping) > 0)
        src_true = bool(fa.live and len(fa.ports) >= int(min_frame_ports))
        tgt_true = bool(fb.live and len(fb.ports) >= int(min_frame_ports))
        true_live = bool(live and src_true and tgt_true)
        transports[ce] = FrameTransport(
            comp_edge=ce,
            source_frame=a,
            target_frame=b,
            n_source_classes=fa.n_classes,
            n_target_classes=fb.n_classes,
            n_source_ports=len(fa.ports),
            n_target_ports=len(fb.ports),
            mapping=mapping,
            accuracy=float(acc),
            coverage_fraction=float(coverage),
            bijective=bool(bij),
            source_entropy_bits=float(h_src),
            live=live,
            true_live=true_live,
        )
    return transports


# ---------------------------------------------------------------------------
# Diamond/path/holonomy analysis
# ---------------------------------------------------------------------------
def orient_cycle_as_two_paths(cycle: Sequence[int], dir_edges: Set[CompEdge]) -> Optional[Tuple[int, int, List[List[int]]]]:
    return ORH.orient_cycle_as_two_paths(cycle, dir_edges)


def path_transport(
    path: Sequence[int],
    transports: Dict[CompEdge, FrameTransport],
    require_true_frames: bool = True,
) -> Tuple[Optional[Dict[int, int]], float, bool, int]:
    """Compose frame transports along a directed observer path.

    When ``require_true_frames`` is true, every edge must be a transport between
    true multi-port frames, not merely between one-port edge labels.
    """
    if len(path) < 2:
        return None, 0.0, False, 0
    first = (int(path[0]), int(path[1]))
    if first not in transports:
        return None, 0.0, False, 0
    first_tr = transports[first]
    if not (first_tr.true_live if require_true_frames else first_tr.live):
        return None, 0.0, False, 0
    n = first_tr.n_source_classes
    cur = {i: i for i in range(n)}
    accs: List[float] = []
    bijs: List[bool] = []
    for i in range(len(path) - 1):
        e = (int(path[i]), int(path[i + 1]))
        tr = transports.get(e)
        if tr is None or not (tr.true_live if require_true_frames else tr.live):
            return None, 0.0, False, n
        cur = compose_maps(cur, tr.mapping)
        accs.append(float(tr.accuracy))
        bijs.append(bool(tr.bijective))
    return cur, float(min(accs) if accs else 0.0), bool(all(bijs)), n


def map_agreement(f: Dict[int, int], g: Dict[int, int], n: int) -> float:
    if n <= 0:
        return 0.0
    ok = 0
    for i in range(n):
        if i in f and i in g and int(f[i]) == int(g[i]):
            ok += 1
    return float(ok / n)


def analyze_frame_cycle(
    cycle: Sequence[int],
    transports: Dict[CompEdge, FrameTransport],
    dir_edges: Set[CompEdge],
    path_agreement_threshold: float = 0.999,
) -> Dict:
    row: Dict[str, object] = dict(
        usable_diamond=0,
        # Backwards-compatible name: flat/path-equal connection.  Gauge scoring
        # should use loop_automorphism_valid/valid_holonomy instead.
        valid_connection=0,
        flat_connection=0,
        valid_holonomy=0,
        loop_automorphism_valid=0,
        nontrivial_holonomy=0,
        loop_automorphism_nontrivial=0,
        source_observer=-1,
        sink_observer=-1,
        path_agreement=0.0,
        path_comparability=0.0,
        path_min_accuracy=0.0,
        path_bijective=0,
        source_frame_classes=0,
        sink_frame_classes=0,
        holonomy_type="not_usable",
        generated_group="none",
        group_order=0,
    )
    oriented = orient_cycle_as_two_paths(cycle, dir_edges)
    if oriented is None:
        return row
    source, sink, paths = oriented
    row.update(usable_diamond=1, source_observer=source, sink_observer=sink,
               path1=" ".join(map(str, paths[0])), path2=" ".join(map(str, paths[1])))
    f1, acc1, bij1, n1 = path_transport(paths[0], transports)
    f2, acc2, bij2, n2 = path_transport(paths[1], transports)
    if f1 is None or f2 is None or n1 <= 0 or n1 != n2:
        row["holonomy_type"] = "missing_transport"
        return row
    # Sink frame size from any final transport.
    last_e = (paths[0][-2], paths[0][-1])
    sink_n = transports[last_e].n_target_classes if last_e in transports else 0
    agree = map_agreement(f1, f2, n1)
    valid_conn = bool(agree >= path_agreement_threshold)
    path_comparable = bool(n1 > 0 and n1 == n2 and f1 is not None and f2 is not None)
    row.update(
        valid_connection=int(valid_conn),
        flat_connection=int(valid_conn),
        path_agreement=float(agree),
        path_comparability=float(min(acc1, acc2)) if path_comparable else 0.0,
        path_min_accuracy=float(min(acc1, acc2)),
        path_bijective=int(bool(bij1 and bij2)),
        source_frame_classes=int(n1),
        sink_frame_classes=int(sink_n),
    )
    if not (bij1 and bij2):
        row["holonomy_type"] = "nonbijective_path"
        return row
    inv_f1 = invert_permutation(f1, n1)
    if inv_f1 is None:
        row["holonomy_type"] = "nonbijective_inverse"
        return row
    delta = compose_maps(f2, inv_f1)  # source frame -> source frame
    if set(delta.keys()) != set(range(n1)) or set(delta.values()) != set(range(n1)):
        row["holonomy_type"] = "nonbijective_delta"
        return row
    htype = permutation_cycle_type(delta, n1)
    g_order, g_name = generated_group_order([delta], n1)
    nontriv = int(htype != "identity")
    row.update(
        valid_holonomy=1,
        loop_automorphism_valid=1,
        nontrivial_holonomy=nontriv,
        loop_automorphism_nontrivial=nontriv,
        holonomy_type=htype,
        generated_group=g_name,
        group_order=int(g_order),
        delta=" ".join(str(delta[i]) for i in range(n1)),
    )
    return row


def _transport_edge_score(tr: Optional[FrameTransport], require_true_frames: bool = True) -> float:
    """Continuous-ish completion score for one observer edge.

    Exact true-live transport receives 1.  Non-live or one-port transport gets 0.
    This deliberately remains a consistency/transport score, not a group target.
    """
    if tr is None:
        return 0.0
    if require_true_frames:
        return 1.0 if tr.true_live else 0.0
    return 1.0 if tr.live else 0.0


def analyze_branch_completion_for_cycle(
    cycle: Sequence[int],
    transports: Dict[CompEdge, FrameTransport],
    dir_edges: Set[CompEdge],
    require_true_frames: bool = True,
) -> Dict[str, float | int]:
    """How close a discovered diamond is to having one/two complete branches."""
    row: Dict[str, float | int] = dict(
        branch_usable_diamond=0,
        best_branch_completion=0.0,
        two_branch_completion=0.0,
        best_complete_branch=0.0,
        two_complete_branches=0.0,
    )
    oriented = orient_cycle_as_two_paths(cycle, dir_edges)
    if oriented is None:
        return row
    _source, _sink, paths = oriented
    row["branch_usable_diamond"] = 1
    path_scores: List[float] = []
    complete_scores: List[float] = []
    for path in paths:
        edges = [(int(path[i]), int(path[i + 1])) for i in range(len(path) - 1)]
        if not edges:
            path_scores.append(0.0)
            complete_scores.append(0.0)
            continue
        scores = [_transport_edge_score(transports.get(e), require_true_frames=require_true_frames) for e in edges]
        path_scores.append(float(np.mean(scores)))
        complete_scores.append(float(min(scores)))
    if path_scores:
        row["best_branch_completion"] = float(max(path_scores))
        row["two_branch_completion"] = float(min(path_scores)) if len(path_scores) >= 2 else 0.0
        row["best_complete_branch"] = float(max(complete_scores))
        row["two_complete_branches"] = float(min(complete_scores)) if len(complete_scores) >= 2 else 0.0
    return row


def global_curvature_audit(
    transports: Dict[CompEdge, FrameTransport],
    require_true_frames: bool = True,
) -> Dict:
    """Gauge-fix a spanning forest and report chord holonomies.

    This is a global obstruction audit.  It does not ask whether one diamond is
    nontrivial in a chosen frame.  It asks whether all live bijective frame
    transports on the observer graph can be flattened simultaneously by local
    frame choices.  Chords of a spanning forest are the gauge-invariant tests.
    """
    usable: Dict[CompEdge, FrameTransport] = {}
    for ce, tr in transports.items():
        live_ok = tr.true_live if require_true_frames else tr.live
        if live_ok and tr.bijective and tr.n_source_classes == tr.n_target_classes and tr.n_source_classes > 0:
            usable[ce] = tr

    nodes: Set[int] = set()
    adj: Dict[int, List[Tuple[int, int, bool]]] = defaultdict(list)
    for (u, v), tr in usable.items():
        nodes.add(int(u)); nodes.add(int(v))
        adj[int(u)].append((int(u), int(v), True))
        adj[int(v)].append((int(u), int(v), False))

    visited: Set[int] = set()
    tree_edges: Set[Tuple[int, int]] = set()
    gauges: Dict[int, Dict[int, int]] = {}
    component_id: Dict[int, int] = {}
    component_n: Dict[int, int] = {}
    n_components = 0

    for root in sorted(nodes):
        if root in visited:
            continue
        n_components += 1
        # Choose the frame size from one incident usable edge.
        incident = [usable[(u, v)] for u, v, _forward in adj[root] if (u, v) in usable]
        if not incident:
            continue
        n = int(incident[0].n_source_classes if incident[0].source_frame == root else incident[0].n_target_classes)
        gauges[root] = {i: i for i in range(n)}  # root frame -> node frame
        component_id[root] = n_components
        component_n[n_components] = n
        visited.add(root)
        dq = deque([root])
        while dq:
            x = dq.popleft()
            for u, v, forward in adj.get(x, []):
                y = v if x == u else u
                tr = usable[(u, v)]
                if tr.n_source_classes != n or tr.n_target_classes != n:
                    continue
                if y in visited:
                    continue
                if forward:
                    # x=u, y=v; gauge[y] = T_uv after gauge[x]
                    gauges[y] = compose_maps(gauges[x], tr.mapping)
                else:
                    # x=v, y=u; gauge[y] = T_uv^{-1} after gauge[x]
                    inv = invert_permutation(tr.mapping, n)
                    if inv is None:
                        continue
                    gauges[y] = compose_maps(gauges[x], inv)
                tree_edges.add(tuple(sorted((u, v))))
                component_id[y] = n_components
                visited.add(y)
                dq.append(y)

    deltas: List[Dict[int, int]] = []
    hol_type_counts: Counter = Counter()
    valid_chords = 0
    nontriv = 0
    chord_count = 0
    for (u, v), tr in usable.items():
        key = tuple(sorted((u, v)))
        if key in tree_edges:
            continue
        if u not in gauges or v not in gauges or component_id.get(u) != component_id.get(v):
            continue
        n = component_n.get(component_id[u], tr.n_source_classes)
        if tr.n_source_classes != n or tr.n_target_classes != n:
            continue
        chord_count += 1
        via_edge = compose_maps(gauges[u], tr.mapping)  # root -> v via u->v
        inv_tree_v = invert_permutation(gauges[v], n)
        if inv_tree_v is None:
            continue
        delta = compose_maps(via_edge, inv_tree_v)      # root -> root
        if set(delta.keys()) != set(range(n)) or set(delta.values()) != set(range(n)):
            continue
        valid_chords += 1
        htype = permutation_cycle_type(delta, n)
        hol_type_counts[htype] += 1
        if htype != "identity":
            nontriv += 1
        deltas.append(delta)

    group_order = 0
    group_name = "none"
    if deltas:
        # All deltas in the first nonempty component have same n by construction
        # for the common control regimes; for mixed components this is a useful
        # conservative group hint rather than a complete classification.
        first_n = len(next(iter(deltas[0].keys()))) if False else max(deltas[0].keys()) + 1
        group_order, group_name = generated_group_order(deltas, first_n)

    return dict(
        global_live_connection_edges=int(len(usable)),
        global_components=int(n_components),
        global_chord_count=int(chord_count),
        global_valid_holonomy=int(valid_chords),
        global_nontrivial_holonomy=int(nontriv),
        global_flat_fraction=float((valid_chords - nontriv) / valid_chords) if valid_chords else 0.0,
        global_group_order=int(group_order),
        global_generated_group=str(group_name),
        global_holonomy_type_counts={str(k): int(v) for k, v in hol_type_counts.items()},
    )


# ---------------------------------------------------------------------------
# Full system analysis
# ---------------------------------------------------------------------------
def observer_cycles_from_system(
    sys: OBG.FiniteRelationalSystem,
    comps: List[List[int]],
    comp_of: Dict[int, int],
    cycle_max_len: int,
) -> Tuple[List[Tuple[int, ...]], Set[CompEdge], Dict[CompEdge, List[Edge]]]:
    pair_edges = directed_pair_edges(sys, comp_of)
    undir: Set[Tuple[int, int]] = set()
    for a, b in pair_edges:
        if a != b:
            undir.add(tuple(sorted((a, b))))
    cycles = OBG.find_undirected_cycles(list(range(len(comps))), sorted(undir), max_len=cycle_max_len)
    return cycles, set(pair_edges.keys()), pair_edges


def analyze_system_bundle(
    sys: OBG.FiniteRelationalSystem,
    graph_id: int = 0,
    max_channel_inputs: int = 256,
    max_channel_backgrounds: int = 256,
    max_state_samples: int = 4096,
    state_warmup: int = 4,
    cycle_max_len: int = 6,
    min_edge_classes: int = 2,
    min_edge_entropy: float = 0.25,
    min_edge_transport: float = 0.95,
    min_frame_classes: int = 2,
    min_frame_entropy: float = 0.25,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    frame_require_live_edge: bool = True,
    min_transport_accuracy: float = 0.999,
    min_transport_coverage: float = 0.999,
    path_agreement_threshold: float = 0.999,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], Dict]:
    """Analyze observer frames, induced connections, and diamond holonomy."""
    if rng is None:
        rng = np.random.default_rng(0)
    edges = sys.edges()
    comps = OBG.tarjan_scc(sys.k, edges)
    comp_of = component_map(comps)
    cycles, dir_edges, pair_edges = observer_cycles_from_system(sys, comps, comp_of, cycle_max_len)

    full_in, full_out = OBG.boundary_edges(set(range(sys.k)), edges)
    full_boundary_violation = int(len(full_in) + len(full_out) != 0)

    edge_q: Dict[CompEdge, ORH.EdgeQuotient] = {}
    edge_rows: List[Dict] = []
    for ce, evs in sorted(pair_edges.items()):
        eq = ORH.edge_common_factor(
            sys,
            ce,
            evs,
            max_channel_inputs,
            max_channel_backgrounds,
            rng,
            min_classes=min_edge_classes,
            min_label_entropy=min_edge_entropy,
            min_transport_mi_norm=min_edge_transport,
        )
        # Soft liveness diagnostic used by blindobserverconnection as a gradient.
        # It is not a group/holonomy target; it only asks whether the boundary
        # channel carries any information from the source side to the target side.
        channel = OBG.estimate_channel_mi(
            sys,
            eq.source_nodes,
            eq.target_nodes,
            max_channel_inputs,
            max_channel_backgrounds,
            rng,
        )
        edge_q[ce] = eq
        edge_rows.append(dict(
            graph_id=graph_id,
            source_observer=ce[0],
            target_observer=ce[1],
            n_micro_edges=len(evs),
            source_nodes=" ".join(map(str, eq.source_nodes)),
            target_nodes=" ".join(map(str, eq.target_nodes)),
            edge_mi_bits=float(channel.get("mi_bits", 0.0)),
            edge_mi_norm=float(channel.get("mi_norm", 0.0)),
            shared_classes=eq.n_classes,
            residual_bits=eq.residual_bits,
            label_entropy_bits=eq.label_entropy_bits,
            transport_mi_norm=eq.transport_mi_norm,
            deterministic_accuracy=eq.deterministic_accuracy,
            live_edge_quotient=int(eq.live),
            micro_edges=" ".join(f"{u}->{v}" for u, v in evs),
        ))

    edge_charts = build_edge_charts(edge_q, mode=frame_coordinate_mode)
    # Add chart-mode diagnostics to the edge rows.
    for row in edge_rows:
        ce = (int(row["source_observer"]), int(row["target_observer"]))
        ch = edge_charts.get(ce)
        if ch is not None:
            row["frame_coordinate_mode"] = ch.mode
            row["chart_classes"] = ch.n_classes
            row["chart_live"] = int(ch.live)
            row["chart_edge_map"] = " ".join(f"{k}->{v}" for k, v in sorted(ch.edge_map.items()))

    samples = warm_state_samples(sys, rng, max_states=max_state_samples, warmup=state_warmup)
    frames = build_observer_frames(
        sys,
        comps,
        edge_charts,
        samples,
        min_frame_classes=min_frame_classes,
        min_frame_entropy=min_frame_entropy,
        require_live_edge=frame_require_live_edge,
        frame_coordinate_mode=frame_coordinate_mode,
    )
    frame_rows: List[Dict] = []
    for oid, fr in sorted(frames.items()):
        port_label_live = bool(fr.live)
        true_frame_live = bool(fr.live and len(fr.ports) >= int(min_frame_ports))
        frame_rows.append(dict(
            graph_id=graph_id,
            observer_id=oid,
            vertices=" ".join(map(str, fr.vertices)),
            n_vertices=len(fr.vertices),
            n_ports=len(fr.ports),
            min_frame_ports=int(min_frame_ports),
            frame_classes=fr.n_classes,
            frame_entropy_bits=fr.entropy_bits,
            port_label_live=int(port_label_live),
            true_frame_live=int(true_frame_live),
            # Backward-compatible column name, now intentionally strict:
            # a live observer frame must coordinate multiple boundary ports.
            frame_live=int(true_frame_live),
            single_port_label_live=int(port_label_live and len(fr.ports) < int(min_frame_ports)),
            coverage_fraction=fr.coverage_fraction,
            ports=";".join(f"{a}->{b}:{side}" for (a, b), side in fr.ports),
        ))

    transports = build_frame_transports(
        frames,
        edge_charts,
        min_accuracy=min_transport_accuracy,
        min_coverage=min_transport_coverage,
        min_frame_ports=min_frame_ports,
    )
    transport_rows: List[Dict] = []
    for ce, tr in sorted(transports.items()):
        transport_rows.append(dict(
            graph_id=graph_id,
            source_observer=ce[0],
            target_observer=ce[1],
            n_source_classes=tr.n_source_classes,
            n_target_classes=tr.n_target_classes,
            n_source_ports=tr.n_source_ports,
            n_target_ports=tr.n_target_ports,
            transport_accuracy=tr.accuracy,
            coverage_fraction=tr.coverage_fraction,
            source_entropy_bits=tr.source_entropy_bits,
            bijective=int(tr.bijective),
            port_label_transport=int(tr.live),
            true_frame_transport=int(tr.true_live),
            # Backward-compatible column name, now strict multi-port transport.
            live_frame_transport=int(tr.true_live),
            mapping=" ".join(f"{k}->{v}" for k, v in sorted(tr.mapping.items())),
        ))

    cycle_rows: List[Dict] = []
    for cidx, cyc in enumerate(cycles):
        row = analyze_frame_cycle(cyc, transports, dir_edges, path_agreement_threshold=path_agreement_threshold)
        row.update(analyze_branch_completion_for_cycle(cyc, transports, dir_edges, require_true_frames=True))
        row.update(
            graph_id=graph_id,
            cycle_id=cidx,
            length=len(cyc),
            observers=" ".join(map(str, cyc)),
        )
        cycle_rows.append(row)

    n_edges = len(edge_rows)
    n_frames = len(frame_rows)
    n_trans = len(transport_rows)
    n_cycles = len(cycle_rows)
    flat_conn = sum(int(r.get("flat_connection", r.get("valid_connection", 0))) for r in cycle_rows)
    valid_conn = flat_conn  # backwards-compatible path-equal connection count
    valid_hol = sum(int(r.get("valid_holonomy", r.get("loop_automorphism_valid", 0))) for r in cycle_rows)
    loop_auto = sum(int(r.get("loop_automorphism_valid", r.get("valid_holonomy", 0))) for r in cycle_rows)
    nontriv = sum(int(r.get("nontrivial_holonomy", r.get("loop_automorphism_nontrivial", 0))) for r in cycle_rows)
    port_label_frames = sum(int(r.get("port_label_live", 0)) for r in frame_rows)
    true_frames = sum(int(r.get("true_frame_live", 0)) for r in frame_rows)
    single_port_frames = sum(int(r.get("single_port_label_live", 0)) for r in frame_rows)
    live_edge_q = sum(int(r.get("live_edge_quotient", 0)) for r in edge_rows)
    port_label_trans = sum(int(r.get("port_label_transport", 0)) for r in transport_rows)
    true_trans = sum(int(r.get("true_frame_transport", 0)) for r in transport_rows)
    global_curv = global_curvature_audit(transports, require_true_frames=True)
    summary = dict(
        graph_id=graph_id,
        q=sys.q,
        n_vertices=sys.k,
        n_scc=len(comps),
        n_feedback_scc=sum(1 for c in comps if len(c) >= 2),
        min_frame_ports=int(min_frame_ports),
        frame_coordinate_mode=str(frame_coordinate_mode),
        n_observer_edges=n_edges,
        n_live_edge_quotients=live_edge_q,
        n_frames=n_frames,
        n_port_label_frames=port_label_frames,
        n_single_port_label_frames=single_port_frames,
        n_live_frames=true_frames,
        n_true_live_frames=true_frames,
        n_frame_transports=n_trans,
        n_port_label_transports=port_label_trans,
        n_live_frame_transports=true_trans,
        n_true_live_frame_transports=true_trans,
        n_observer_cycles=n_cycles,
        n_usable_diamonds=sum(int(r.get("usable_diamond", 0)) for r in cycle_rows),
        # Flat/path-equal connection counts are kept separately from
        # automorphism-valid loop closure.  A gauge connection may have
        # non-identity delta and therefore need not be path-equal.
        n_valid_connections=valid_conn,
        n_flat_connections=flat_conn,
        n_valid_holonomy=valid_hol,
        n_loop_automorphism_valid=loop_auto,
        n_nontrivial_holonomy=nontriv,
        max_path_agreement=float(max([float(r.get("path_agreement", 0.0)) for r in cycle_rows], default=0.0)),
        max_path_comparability=float(max([float(r.get("path_comparability", 0.0)) for r in cycle_rows], default=0.0)),
        max_path_bijective=float(max([float(r.get("path_bijective", 0.0)) for r in cycle_rows], default=0.0)),
        max_loop_automorphism_validity=float(max([float(r.get("loop_automorphism_valid", r.get("valid_holonomy", 0))) for r in cycle_rows], default=0.0)),
        max_branch_completion=float(max([float(r.get("best_branch_completion", 0.0)) for r in cycle_rows], default=0.0)),
        max_two_branch_completion=float(max([float(r.get("two_branch_completion", 0.0)) for r in cycle_rows], default=0.0)),
        max_complete_branch=float(max([float(r.get("best_complete_branch", 0.0)) for r in cycle_rows], default=0.0)),
        max_two_complete_branches=float(max([float(r.get("two_complete_branches", 0.0)) for r in cycle_rows], default=0.0)),
        mean_frame_classes=float(np.mean([r["frame_classes"] for r in frame_rows])) if frame_rows else 0.0,
        mean_edge_shared_classes=float(np.mean([r["shared_classes"] for r in edge_rows])) if edge_rows else 0.0,
        mean_edge_mi_bits=float(np.mean([r["edge_mi_bits"] for r in edge_rows])) if edge_rows else 0.0,
        mean_edge_mi_norm=float(np.mean([r["edge_mi_norm"] for r in edge_rows])) if edge_rows else 0.0,
        port_label_frame_fraction=float(port_label_frames / n_frames) if n_frames else 0.0,
        live_frame_fraction=float(true_frames / n_frames) if n_frames else 0.0,
        true_live_frame_fraction=float(true_frames / n_frames) if n_frames else 0.0,
        single_port_label_frame_fraction=float(single_port_frames / n_frames) if n_frames else 0.0,
        live_edge_quotient_fraction=float(live_edge_q / n_edges) if n_edges else 0.0,
        port_label_transport_fraction=float(port_label_trans / n_trans) if n_trans else 0.0,
        live_frame_transport_fraction=float(true_trans / n_trans) if n_trans else 0.0,
        true_live_frame_transport_fraction=float(true_trans / n_trans) if n_trans else 0.0,
        valid_connection_fraction=float(valid_conn / n_cycles) if n_cycles else 0.0,
        flat_connection_fraction=float(flat_conn / n_cycles) if n_cycles else 0.0,
        valid_holonomy_fraction=float(valid_hol / n_cycles) if n_cycles else 0.0,
        loop_automorphism_valid_fraction=float(loop_auto / n_cycles) if n_cycles else 0.0,
        nontrivial_holonomy_fraction=float(nontriv / valid_hol) if valid_hol else 0.0,
        full_boundary_violation=full_boundary_violation,
        **global_curv,
    )
    return frame_rows, edge_rows, transport_rows, cycle_rows, summary


# ---------------------------------------------------------------------------
# Experiment runner / CLI
# ---------------------------------------------------------------------------
def make_system_for_mode(
    q: int,
    vertices: int,
    graph_mode: str,
    components: int,
    edge_prob: float,
    inter_prob: float,
    extra_intra_prob: float,
    max_pred: Optional[int],
    rng: np.random.Generator,
) -> OBG.FiniteRelationalSystem:
    mode = graph_mode.lower()
    if mode == "er":
        return OBG.make_random_system(q, vertices, edge_prob, rng, max_pred=max_pred)
    if mode == "componented":
        return OBG.make_componented_system(q, vertices, components, inter_prob, extra_intra_prob, rng, max_pred=max_pred)
    if mode == "frame_flat_c2":
        return make_frame_diamond_control(q=q, twist=False)
    if mode == "frame_twist_c2":
        return make_frame_diamond_control(q=q, twist=True)
    if mode == "frame_random_diamond":
        return make_random_frame_diamond(q=q, rng=rng, max_pred=max_pred)
    if mode == "frame_theta_flat_c2":
        return make_frame_theta_control(q=q, twists=(False, False, False))
    if mode == "frame_theta_twist_c2":
        return make_frame_theta_control(q=q, twists=(False, False, True))
    if mode == "frame_random_theta":
        return make_random_frame_theta(q=q, rng=rng, max_pred=max_pred)
    if mode == "frame_double_flat_c2":
        return make_frame_double_diamond_control(q=q, twists=(False, False))
    if mode == "frame_double_twist_c2":
        return make_frame_double_diamond_control(q=q, twists=(True, False))
    if mode == "frame_random_double_diamond":
        return make_random_frame_double_diamond(q=q, rng=rng, max_pred=max_pred)
    raise ValueError(f"unknown graph mode: {graph_mode}")


def run_audit(
    q: int = 4,
    graphs: int = 20,
    vertices: int = 18,
    graph_mode: str = "componented",
    components: int = 5,
    edge_prob: float = 0.16,
    inter_prob: float = 0.35,
    extra_intra_prob: float = 0.25,
    max_pred: Optional[int] = 4,
    max_channel_inputs: int = 256,
    max_channel_backgrounds: int = 256,
    max_state_samples: int = 4096,
    state_warmup: int = 4,
    cycle_max_len: int = 6,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[object, object, object, object, Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for run_audit/CLI output")
    rng0 = np.random.default_rng(seed)
    all_frames: List[Dict] = []
    all_edges: List[Dict] = []
    all_trans: List[Dict] = []
    all_cycles: List[Dict] = []
    summaries: List[Dict] = []
    for gi in range(graphs):
        grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
        sys = make_system_for_mode(q, vertices, graph_mode, components, edge_prob, inter_prob, extra_intra_prob, max_pred, grng)
        fr, er, tr, cr, sm = analyze_system_bundle(
            sys,
            graph_id=gi,
            max_channel_inputs=max_channel_inputs,
            max_channel_backgrounds=max_channel_backgrounds,
            max_state_samples=max_state_samples,
            state_warmup=state_warmup,
            cycle_max_len=cycle_max_len,
            min_frame_ports=min_frame_ports,
            frame_coordinate_mode=frame_coordinate_mode,
            rng=grng,
        )
        all_frames.extend(fr)
        all_edges.extend(er)
        all_trans.extend(tr)
        all_cycles.extend(cr)
        summaries.append(sm)
        if verbose:
            print(
                f"observer-frame graph={gi} frames={sm['n_live_frames']}/{sm['n_frames']} "
                f"trans={sm['n_live_frame_transports']}/{sm['n_frame_transports']} "
                f"conn={sm['n_valid_connections']} hol={sm['n_valid_holonomy']} nontriv={sm['n_nontrivial_holonomy']}",
                flush=True,
            )

    frame_df = pd.DataFrame(all_frames)
    edge_df = pd.DataFrame(all_edges)
    trans_df = pd.DataFrame(all_trans)
    cycle_df = pd.DataFrame(all_cycles)
    summ_df = pd.DataFrame(summaries)

    total_valid_conn = int(summ_df["n_valid_connections"].sum()) if len(summ_df) else 0
    total_valid_hol = int(summ_df["n_valid_holonomy"].sum()) if len(summ_df) else 0
    total_nontriv = int(summ_df["n_nontrivial_holonomy"].sum()) if len(summ_df) else 0
    total_live_trans = int(summ_df["n_live_frame_transports"].sum()) if len(summ_df) else 0
    total_live_frames = int(summ_df["n_live_frames"].sum()) if len(summ_df) else 0
    total_port_label_frames = int(summ_df["n_port_label_frames"].sum()) if len(summ_df) and "n_port_label_frames" in summ_df else total_live_frames
    total_single_port_label_frames = int(summ_df["n_single_port_label_frames"].sum()) if len(summ_df) and "n_single_port_label_frames" in summ_df else 0
    total_global_valid_hol = int(summ_df["global_valid_holonomy"].sum()) if len(summ_df) and "global_valid_holonomy" in summ_df else 0
    total_global_nontriv = int(summ_df["global_nontrivial_holonomy"].sum()) if len(summ_df) and "global_nontrivial_holonomy" in summ_df else 0

    if total_nontriv > 0 or total_global_nontriv > 0:
        verdict = "OBSERVER-FRAME NONTRIVIAL HOLONOMY SIGNAL: observer frames carry nontrivial loop automorphisms"
    elif total_valid_hol > 0 or total_global_valid_hol > 0:
        verdict = "OBSERVER-FRAME AUTOMORPHISM-VALID CONNECTION SIGNAL: extracted frames close as loop automorphisms; observed holonomy may be flat"
    elif total_valid_conn > 0:
        verdict = "OBSERVER-FRAME CONNECTION SIGNAL: path-consistent frame transports appear"
    elif total_live_trans > 0:
        verdict = "OBSERVER-FRAME TRANSPORT SIGNAL: live frame transports appear but no path-consistent diamond was found"
    elif total_live_frames > 0:
        verdict = "OBSERVER-FRAME SIGNAL: live observer boundary frames appear but connection is weak"
    else:
        verdict = "NO OBSERVER-FRAME CONNECTION SIGNAL in this regime"

    group_counts: Dict[str, int] = {}
    hol_type_counts: Dict[str, int] = {}
    if len(cycle_df) and "valid_holonomy" in cycle_df:
        valid = cycle_df[cycle_df["valid_holonomy"] == 1]
        if len(valid):
            group_counts = {str(k): int(v) for k, v in Counter(valid["generated_group"].astype(str)).items()}
            hol_type_counts = {str(k): int(v) for k, v in Counter(valid["holonomy_type"].astype(str)).items()}

    summary = dict(
        verdict=verdict,
        q=q,
        graph_mode=graph_mode,
        n_graphs=int(graphs),
        n_vertices=int(vertices),
        frame_coordinate_mode=str(frame_coordinate_mode),
        total_port_label_frames=total_port_label_frames,
        total_single_port_label_frames=total_single_port_label_frames,
        total_live_frames=total_live_frames,
        total_true_live_frames=total_live_frames,
        total_live_frame_transports=total_live_trans,
        total_true_live_frame_transports=total_live_trans,
        total_valid_connections=total_valid_conn,
        total_flat_connections=total_valid_conn,
        total_valid_holonomy=total_valid_hol,
        total_loop_automorphism_valid=total_valid_hol,
        total_nontrivial_holonomy=total_nontriv,
        total_global_valid_holonomy=total_global_valid_hol,
        total_global_nontrivial_holonomy=total_global_nontriv,
        graph_fraction_with_port_label_frames=float((summ_df["n_port_label_frames"] > 0).mean()) if len(summ_df) and "n_port_label_frames" in summ_df else 0.0,
        graph_fraction_with_live_frames=float((summ_df["n_live_frames"] > 0).mean()) if len(summ_df) else 0.0,
        graph_fraction_with_connection=float((summ_df["n_valid_connections"] > 0).mean()) if len(summ_df) else 0.0,
        graph_fraction_with_holonomy=float((summ_df["n_valid_holonomy"] > 0).mean()) if len(summ_df) else 0.0,
        graph_fraction_with_nontrivial_holonomy=float((summ_df["n_nontrivial_holonomy"] > 0).mean()) if len(summ_df) else 0.0,
        mean_port_label_frame_fraction=float(summ_df["port_label_frame_fraction"].mean()) if len(summ_df) and "port_label_frame_fraction" in summ_df else 0.0,
        mean_single_port_label_frame_fraction=float(summ_df["single_port_label_frame_fraction"].mean()) if len(summ_df) and "single_port_label_frame_fraction" in summ_df else 0.0,
        mean_live_frame_fraction=float(summ_df["live_frame_fraction"].mean()) if len(summ_df) else 0.0,
        mean_true_live_frame_fraction=float(summ_df["true_live_frame_fraction"].mean()) if len(summ_df) and "true_live_frame_fraction" in summ_df else 0.0,
        mean_live_edge_quotient_fraction=float(summ_df["live_edge_quotient_fraction"].mean()) if len(summ_df) else 0.0,
        mean_edge_mi_bits=float(summ_df["mean_edge_mi_bits"].mean()) if len(summ_df) else 0.0,
        mean_edge_mi_norm=float(summ_df["mean_edge_mi_norm"].mean()) if len(summ_df) else 0.0,
        mean_port_label_transport_fraction=float(summ_df["port_label_transport_fraction"].mean()) if len(summ_df) and "port_label_transport_fraction" in summ_df else 0.0,
        mean_live_frame_transport_fraction=float(summ_df["live_frame_transport_fraction"].mean()) if len(summ_df) else 0.0,
        mean_true_live_frame_transport_fraction=float(summ_df["true_live_frame_transport_fraction"].mean()) if len(summ_df) and "true_live_frame_transport_fraction" in summ_df else 0.0,
        mean_max_branch_completion=float(summ_df["max_branch_completion"].mean()) if len(summ_df) and "max_branch_completion" in summ_df else 0.0,
        mean_max_two_branch_completion=float(summ_df["max_two_branch_completion"].mean()) if len(summ_df) and "max_two_branch_completion" in summ_df else 0.0,
        mean_max_complete_branch=float(summ_df["max_complete_branch"].mean()) if len(summ_df) and "max_complete_branch" in summ_df else 0.0,
        mean_max_two_complete_branches=float(summ_df["max_two_complete_branches"].mean()) if len(summ_df) and "max_two_complete_branches" in summ_df else 0.0,
        mean_max_path_comparability=float(summ_df["max_path_comparability"].mean()) if len(summ_df) and "max_path_comparability" in summ_df else 0.0,
        mean_max_loop_automorphism_validity=float(summ_df["max_loop_automorphism_validity"].mean()) if len(summ_df) and "max_loop_automorphism_validity" in summ_df else 0.0,
        max_branch_completion=float(summ_df["max_branch_completion"].max()) if len(summ_df) and "max_branch_completion" in summ_df else 0.0,
        max_two_branch_completion=float(summ_df["max_two_branch_completion"].max()) if len(summ_df) and "max_two_branch_completion" in summ_df else 0.0,
        max_complete_branch=float(summ_df["max_complete_branch"].max()) if len(summ_df) and "max_complete_branch" in summ_df else 0.0,
        max_two_complete_branches=float(summ_df["max_two_complete_branches"].max()) if len(summ_df) and "max_two_complete_branches" in summ_df else 0.0,
        max_path_comparability=float(summ_df["max_path_comparability"].max()) if len(summ_df) and "max_path_comparability" in summ_df else 0.0,
        max_loop_automorphism_validity=float(summ_df["max_loop_automorphism_validity"].max()) if len(summ_df) and "max_loop_automorphism_validity" in summ_df else 0.0,
        mean_valid_connection_fraction=float(summ_df["valid_connection_fraction"].mean()) if len(summ_df) else 0.0,
        mean_valid_holonomy_fraction=float(summ_df["valid_holonomy_fraction"].mean()) if len(summ_df) else 0.0,
        mean_loop_automorphism_valid_fraction=float(summ_df["loop_automorphism_valid_fraction"].mean()) if len(summ_df) and "loop_automorphism_valid_fraction" in summ_df else 0.0,
        mean_max_path_agreement=float(summ_df["max_path_agreement"].mean()) if len(summ_df) else 0.0,
        mean_global_chord_count=float(summ_df["global_chord_count"].mean()) if len(summ_df) and "global_chord_count" in summ_df else 0.0,
        global_valid_holonomy=total_global_valid_hol,
        global_nontrivial_holonomy=total_global_nontriv,
        graph_fraction_with_global_holonomy=float((summ_df["global_valid_holonomy"] > 0).mean()) if len(summ_df) and "global_valid_holonomy" in summ_df else 0.0,
        graph_fraction_with_global_nontrivial_holonomy=float((summ_df["global_nontrivial_holonomy"] > 0).mean()) if len(summ_df) and "global_nontrivial_holonomy" in summ_df else 0.0,
        group_counts=group_counts,
        holonomy_type_counts=hol_type_counts,
        full_boundary_violations=int(summ_df["full_boundary_violation"].sum()) if len(summ_df) else 0,
    )
    return frame_df, edge_df, trans_df, cycle_df, summary


def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def write_outputs(frame_df, edge_df, trans_df, cycle_df, summary: Dict, out: str) -> None:
    frame_df.to_csv(out, index=False)
    edge_df.to_csv(derived_path(out, "_edges"), index=False)
    trans_df.to_csv(derived_path(out, "_transports"), index=False)
    cycle_df.to_csv(derived_path(out, "_cycles"), index=False)
    with open(derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def make_plot(frame_df, trans_df, cycle_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(summary.get("verdict", "observer frame bundle"), fontsize=11)
    if len(frame_df) and "frame_classes" in frame_df:
        axes[0].hist(frame_df["frame_classes"].dropna().values, bins=20)
    axes[0].set_title("frame classes")
    axes[0].set_xlabel("classes")
    axes[0].set_ylabel("count")

    if len(trans_df) and "transport_accuracy" in trans_df:
        axes[1].hist(trans_df["transport_accuracy"].dropna().values, bins=20)
    axes[1].set_title("frame transport accuracy")
    axes[1].set_xlabel("accuracy")

    if len(cycle_df) and "loop_automorphism_valid" in cycle_df:
        axes[2].hist(cycle_df["loop_automorphism_valid"].dropna().values, bins=3)
        axes[2].set_title("loop automorphism valid")
        axes[2].set_xlabel("0/1")
    elif len(cycle_df) and "path_agreement" in cycle_df:
        axes[2].hist(cycle_df["path_agreement"].dropna().values, bins=20)
        axes[2].set_title("diamond path agreement")
        axes[2].set_xlabel("agreement")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Observer-frame bundle audit")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--graphs", type=int, default=20)
    p.add_argument("--vertices", type=int, default=18)
    p.add_argument("--graph-mode", choices=["componented", "er", "frame_flat_c2", "frame_twist_c2", "frame_random_diamond", "frame_theta_flat_c2", "frame_theta_twist_c2", "frame_random_theta", "frame_double_flat_c2", "frame_double_twist_c2", "frame_random_double_diamond"], default="componented")
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.16)
    p.add_argument("--inter-prob", type=float, default=0.35)
    p.add_argument("--extra-intra-prob", type=float, default=0.25)
    p.add_argument("--max-pred", type=int, default=4)
    p.add_argument("--max-channel-inputs", type=int, default=256)
    p.add_argument("--max-channel-backgrounds", type=int, default=256)
    p.add_argument("--max-state-samples", type=int, default=4096)
    p.add_argument("--state-warmup", type=int, default=4)
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--min-frame-ports", type=int, default=2, help="minimum incident ports for a true observer frame")
    p.add_argument("--frame-coordinate-mode", choices=["local_charts", "common_factor"], default="local_charts",
                   help="local_charts keeps oriented endpoint transition maps; common_factor reproduces shared equalizer coordinates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/observer_frame_bundle.csv")
    p.add_argument("--plot", default="example_results/fig_observer_frame_bundle.png")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    max_pred = None if args.max_pred is None or args.max_pred <= 0 else int(args.max_pred)
    frame_df, edge_df, trans_df, cycle_df, summary = run_audit(
        q=args.q,
        graphs=args.graphs,
        vertices=args.vertices,
        graph_mode=args.graph_mode,
        components=args.components,
        edge_prob=args.edge_prob,
        inter_prob=args.inter_prob,
        extra_intra_prob=args.extra_intra_prob,
        max_pred=max_pred,
        max_channel_inputs=args.max_channel_inputs,
        max_channel_backgrounds=args.max_channel_backgrounds,
        max_state_samples=args.max_state_samples,
        state_warmup=args.state_warmup,
        cycle_max_len=args.cycle_max_len,
        min_frame_ports=args.min_frame_ports,
        frame_coordinate_mode=args.frame_coordinate_mode,
        seed=args.seed,
        verbose=not args.quiet,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_outputs(frame_df, edge_df, trans_df, cycle_df, summary, args.out)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plot(frame_df, trans_df, cycle_df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {derived_path(args.out, '_edges')}")
    print(f"wrote {derived_path(args.out, '_transports')}")
    print(f"wrote {derived_path(args.out, '_cycles')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
