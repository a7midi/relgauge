"""
observersectiondynamics.py

Section-obstruction dynamics audit for observer-frame connections.

The observer-frame bundle layer extracts local observer frames F_O and oriented
edge transports U_e:F_O->F_P.  This module asks the next question without adding
matter, particles, flux, lifetime, or motion by hand:

    Given a selected/certified observer-frame connection, what obstruction
    pattern is induced by the microscopic states themselves, and does that
    pattern have a schedule-invariant recurrent quotient?

For a microscopic state x and a live frame transport edge e:O->P, the module
reads the endpoint-local frame value on O's source port and P's target port.
The section-gluing residual is

    r_e(x)=0  if  s_P,e(x) = U_e(s_O,e(x))
           1  otherwise.

The residual vector r(x) is then evolved under sampled admissible update
schedules.  Schedule order is treated as gauge: the module computes the quotient
of residual patterns forced by schedule ambiguity and reports whether the
resulting quotient is nontrivial and recurrent.

A single nontrivial C2 diamond should show static frustration but not moving
defects.  Multi-cycle arenas such as theta or two coupled diamonds are the first
places where residual representatives can potentially shift between loops.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import observerboundarygeometry as OBG
    from . import observerframebundle as OBF
    from . import observerrelativeholonomy as ORH
except Exception:  # pragma: no cover
    import observerboundarygeometry as OBG  # type: ignore
    import observerframebundle as OBF  # type: ignore
    import observerrelativeholonomy as ORH  # type: ignore

CompEdge = Tuple[int, int]


# ---------------------------------------------------------------------------
# System loading / schedules
# ---------------------------------------------------------------------------
def clone_system(sys: OBG.FiniteRelationalSystem) -> OBG.FiniteRelationalSystem:
    return OBG.FiniteRelationalSystem(q=int(sys.q), preds=[tuple(ps) for ps in sys.preds], tables=[np.array(t, copy=True) for t in sys.tables])


def load_system_from_winners(path: str, index: int = 0) -> OBG.FiniteRelationalSystem:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "system" in payload:
        return clone_system(payload["system"])
    if not isinstance(payload, list):
        raise ValueError("winners pickle must contain a list of candidate payloads or a single candidate dict")
    if not payload:
        raise ValueError("empty winners pickle")
    item = payload[int(index)]
    if isinstance(item, dict) and "system" in item:
        return clone_system(item["system"])
    if hasattr(item, "system"):
        return clone_system(item.system)
    raise ValueError("winner item has no system")


def make_system_from_args(args: argparse.Namespace, rng: np.random.Generator) -> OBG.FiniteRelationalSystem:
    max_pred = None if args.max_pred is None or int(args.max_pred) <= 0 else int(args.max_pred)
    if args.winners:
        return load_system_from_winners(args.winners, args.winner_index)
    return OBF.make_system_for_mode(
        q=int(args.q),
        vertices=int(args.vertices),
        graph_mode=str(args.graph_mode),
        components=int(args.components),
        edge_prob=float(args.edge_prob),
        inter_prob=float(args.inter_prob),
        extra_intra_prob=float(args.extra_intra_prob),
        max_pred=max_pred,
        rng=rng,
    )


def sequential_step(sys: OBG.FiniteRelationalSystem, state: Sequence[int], order: Sequence[int]) -> Tuple[int, ...]:
    cur = [int(x) for x in state]
    for v in order:
        cur[int(v)] = int(sys.eval_vertex(int(v), cur))
    return tuple(cur)


def schedule_orders(k: int, schedule_samples: int, rng: np.random.Generator, include_parallel_marker: bool = True) -> List[Tuple[int, ...]]:
    orders: List[Tuple[int, ...]] = []
    if k <= 7 and schedule_samples <= 0:
        orders = [tuple(p) for p in itertools.permutations(range(k))]
    else:
        n = max(1, int(schedule_samples))
        seen: Set[Tuple[int, ...]] = set()
        ident = tuple(range(k)); rev = tuple(reversed(range(k)))
        for o in (ident, rev):
            seen.add(o); orders.append(o)
        while len(orders) < n:
            o = tuple(int(x) for x in rng.permutation(k))
            if o not in seen:
                seen.add(o); orders.append(o)
    return orders


# ---------------------------------------------------------------------------
# Bundle reconstruction
# ---------------------------------------------------------------------------
@dataclass
class BundleObjects:
    comps: List[List[int]]
    frames: Dict[int, OBF.ObserverFrame]
    edge_charts: Dict[CompEdge, OBF.EdgeChart]
    transports: Dict[CompEdge, OBF.FrameTransport]
    cycles: List[Tuple[int, ...]]
    dir_edges: Set[CompEdge]
    summary: Dict


def build_bundle_objects(
    sys: OBG.FiniteRelationalSystem,
    rng: np.random.Generator,
    max_channel_inputs: int = 256,
    max_channel_backgrounds: int = 256,
    max_state_samples: int = 4096,
    state_warmup: int = 4,
    cycle_max_len: int = 6,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
) -> BundleObjects:
    comps = OBG.tarjan_scc(sys.k, sys.edges())
    comp_of = OBF.component_map(comps)
    cycles, dir_edges, pair_edges = OBF.observer_cycles_from_system(sys, comps, comp_of, cycle_max_len)
    edge_q: Dict[CompEdge, ORH.EdgeQuotient] = {}
    for ce, evs in pair_edges.items():
        edge_q[ce] = ORH.edge_common_factor(sys, ce, evs, max_channel_inputs, max_channel_backgrounds, rng)
    edge_charts = OBF.build_edge_charts(edge_q, mode=frame_coordinate_mode)
    state_samples = OBF.warm_state_samples(sys, rng, max_states=max_state_samples, warmup=state_warmup)
    frames = OBF.build_observer_frames(
        sys, comps, edge_charts, state_samples,
        min_frame_classes=2, min_frame_entropy=0.25,
        require_live_edge=True, frame_coordinate_mode=frame_coordinate_mode,
    )
    transports = OBF.build_frame_transports(frames, edge_charts, min_frame_ports=min_frame_ports)
    _fr, _er, _tr, _cr, summary = OBF.analyze_system_bundle(
        sys, graph_id=0,
        max_channel_inputs=max_channel_inputs,
        max_channel_backgrounds=max_channel_backgrounds,
        max_state_samples=max_state_samples,
        state_warmup=state_warmup,
        cycle_max_len=cycle_max_len,
        min_frame_ports=min_frame_ports,
        frame_coordinate_mode=frame_coordinate_mode,
        rng=rng,
    )
    return BundleObjects(comps, frames, edge_charts, transports, cycles, dir_edges, summary)


def frame_value_for_port(frame: OBF.ObserverFrame, port: Tuple[CompEdge, str], raw_label: Optional[int]) -> Optional[int]:
    if raw_label is None:
        return None
    val = frame.port_label_to_frame.get((port, int(raw_label)))
    return None if val is None else int(val)


def live_transport_edges(bundle: BundleObjects, require_true: bool = True) -> List[CompEdge]:
    edges: List[CompEdge] = []
    for ce, tr in sorted(bundle.transports.items()):
        ok = bool(tr.true_live if require_true else tr.live)
        if ok and tr.bijective and tr.n_source_classes == tr.n_target_classes and tr.n_source_classes > 0:
            edges.append(ce)
    return edges


# ---------------------------------------------------------------------------
# Residual patterns and quotients
# ---------------------------------------------------------------------------
def residual_vector(
    sys: OBG.FiniteRelationalSystem,
    state: Sequence[int],
    bundle: BundleObjects,
    edges: Sequence[CompEdge],
) -> Tuple[int, ...]:
    bits: List[int] = []
    for ce in edges:
        tr = bundle.transports[ce]
        chart = bundle.edge_charts[ce]
        u, v = ce
        src_frame = bundle.frames[u]
        tgt_frame = bundle.frames[v]
        src_lab = OBF.port_side_label(chart, "source", state, sys.q)
        tgt_lab = OBF.port_side_label(chart, "target", state, sys.q)
        src_val = frame_value_for_port(src_frame, (ce, "source"), src_lab)
        tgt_val = frame_value_for_port(tgt_frame, (ce, "target"), tgt_lab)
        if src_val is None or tgt_val is None or src_val not in tr.mapping:
            bits.append(1)
        else:
            bits.append(0 if int(tr.mapping[int(src_val)]) == int(tgt_val) else 1)
    return tuple(bits)


def residual_code(bits: Sequence[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return int(out)


def code_bits(code: int, n: int) -> Tuple[int, ...]:
    return tuple((int(code) >> (n - 1 - i)) & 1 for i in range(n))


def observer_frame_coherence(
    sys: OBG.FiniteRelationalSystem,
    state: Sequence[int],
    bundle: BundleObjects,
) -> Tuple[int, int, int]:
    """Return coherent, incoherent, unresolved frame counts for a state."""
    coherent = incoherent = unresolved = 0
    for oid, frame in bundle.frames.items():
        if not frame.live or len(frame.ports) == 0:
            continue
        vals: List[int] = []
        for port in frame.ports:
            ce, side = port
            chart = bundle.edge_charts.get(ce)
            if chart is None:
                continue
            raw = OBF.port_side_label(chart, side, state, sys.q)
            val = frame_value_for_port(frame, port, raw)
            if val is not None:
                vals.append(int(val))
        if not vals:
            unresolved += 1
        elif len(set(vals)) == 1:
            coherent += 1
        else:
            incoherent += 1
    return coherent, incoherent, unresolved


class UnionFind:
    def __init__(self, xs: Iterable[int]):
        vals = sorted(set(int(x) for x in xs))
        self.parent = {x: x for x in vals}
        self.rank = {x: 0 for x in vals}

    def find(self, x: int) -> int:
        x = int(x)
        if x not in self.parent:
            self.parent[x] = x; self.rank[x] = 0
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


def deterministic_schedule_quotient(nodes: Set[int], edges: Dict[int, Set[int]]) -> Dict[int, int]:
    """Coarsen residual states until schedule ambiguity has one successor block.

    Starting from the discrete residual-pattern partition, if a block has
    successors in multiple blocks, those successor blocks are identified.  The
    fixed point is the finest quotient forced by this deterministic-transition
    consistency requirement.  It may collapse to one block.
    """
    uf = UnionFind(nodes)
    changed = True
    while changed:
        changed = False
        blocks: Dict[int, List[int]] = defaultdict(list)
        for n in nodes:
            blocks[uf.find(n)].append(n)
        for members in blocks.values():
            succ_roots = sorted({uf.find(s) for m in members for s in edges.get(m, set())})
            if len(succ_roots) > 1:
                first = succ_roots[0]
                for r in succ_roots[1:]:
                    if uf.find(first) != uf.find(r):
                        uf.union(first, r); changed = True
    roots = sorted({uf.find(n) for n in nodes})
    root_to_q = {r: i for i, r in enumerate(roots)}
    return {int(n): int(root_to_q[uf.find(n)]) for n in nodes}


def recurrent_nodes(nodes: Set[int], edges: Dict[int, Set[int]]) -> Set[int]:
    idx = 0
    stack: List[int] = []
    onstack: Set[int] = set()
    indices: Dict[int, int] = {}
    low: Dict[int, int] = {}
    out: Set[int] = set()

    def strong(v: int) -> None:
        nonlocal idx
        indices[v] = idx; low[v] = idx; idx += 1
        stack.append(v); onstack.add(v)
        for w in edges.get(v, set()):
            if w not in indices:
                strong(w); low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop(); onstack.remove(w); comp.append(w)
                if w == v: break
            if len(comp) > 1 or (len(comp) == 1 and comp[0] in edges.get(comp[0], set())):
                out.update(comp)

    for v in sorted(nodes):
        if v not in indices:
            strong(v)
    return out


def entropy_counts(counts: Iterable[int]) -> float:
    arr = np.asarray([float(c) for c in counts if c > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# V2: coherent sections, cycle syndromes, and cohomology representatives
# ---------------------------------------------------------------------------
@dataclass
class CycleBasis:
    nodes: Tuple[int, ...]
    edges: Tuple[CompEdge, ...]
    basis: Tuple[Tuple[int, ...], ...]
    chords: Tuple[CompEdge, ...]

    @property
    def beta1(self) -> int:
        return len(self.basis)


def _path_edge_indices(tree_adj: Dict[int, List[Tuple[int, int]]], src: int, dst: int) -> Optional[List[int]]:
    dq = deque([(int(src), [])])
    seen = {int(src)}
    while dq:
        v, path = dq.popleft()
        if v == int(dst):
            return path
        for nb, ei in tree_adj.get(v, []):
            if nb in seen:
                continue
            seen.add(nb)
            dq.append((nb, path + [int(ei)]))
    return None


def cycle_basis_for_edges(edges: Sequence[CompEdge]) -> CycleBasis:
    """Build a deterministic Z2 cycle basis from live observer-frame edges.

    The basis is obtained by scanning edges in order, adding spanning-forest
    edges when they connect components, and recording one chord cycle for every
    remaining edge.  For Z2 section residuals, orientation is irrelevant; each
    basis row is the set of edge indices whose parity defines a syndrome bit.
    """
    edges_t = tuple((int(u), int(v)) for u, v in edges)
    nodes = tuple(sorted({x for e in edges_t for x in e}))
    uf = UnionFind(nodes)
    tree_adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    basis: List[Tuple[int, ...]] = []
    chords: List[CompEdge] = []
    for ei, (u, v) in enumerate(edges_t):
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            tree_adj[u].append((v, ei))
            tree_adj[v].append((u, ei))
        else:
            path = _path_edge_indices(tree_adj, u, v)
            if path is None:
                continue
            cyc = tuple(sorted(set(path + [ei])))
            basis.append(cyc)
            chords.append((u, v))
    return CycleBasis(nodes=nodes, edges=edges_t, basis=tuple(basis), chords=tuple(chords))


def binary_edge_connection_bits(bundle: BundleObjects, edges: Sequence[CompEdge]) -> Optional[Tuple[int, ...]]:
    """Return C2 edge transport bits for binary bijective transports, else None."""
    bits: List[int] = []
    for ce in edges:
        tr = bundle.transports.get(ce)
        if tr is None or tr.n_source_classes != 2 or tr.n_target_classes != 2:
            return None
        m = {int(k): int(v) for k, v in tr.mapping.items()}
        if set(m.keys()) != {0, 1} or set(m.values()) != {0, 1}:
            return None
        if m == {0: 0, 1: 1}:
            bits.append(0)
        elif m == {0: 1, 1: 0}:
            bits.append(1)
        else:
            return None
    return tuple(bits)


def syndrome_bits(bits: Sequence[int], basis: CycleBasis) -> Tuple[int, ...]:
    out: List[int] = []
    for row in basis.basis:
        b = 0
        for ei in row:
            if 0 <= int(ei) < len(bits):
                b ^= int(bits[int(ei)]) & 1
        out.append(b)
    return tuple(out)


def _code_to_bit_string(code: int, n_bits: int, invalid_code: Optional[int] = None) -> str:
    if invalid_code is not None and int(code) == int(invalid_code):
        return "INVALID"
    if n_bits <= 0:
        return ""
    return "".join(str(x) for x in code_bits(int(code), int(n_bits)))


def observer_section_values(
    sys: OBG.FiniteRelationalSystem,
    state: Sequence[int],
    bundle: BundleObjects,
    observer_ids: Sequence[int],
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Read one coherent frame value per observer, when possible.

    A coherent section value exists for an observer only when all readable live
    port charts in its frame agree on the same frame coordinate.  This is the
    key separation between genuine section data and the older port-local
    residual patterns: if an observer's own frame is internally inconsistent,
    edge-gluing residuals are not yet section obstructions.
    """
    values: Dict[int, int] = {}
    status: Dict[int, str] = {}
    for oid in observer_ids:
        frame = bundle.frames.get(int(oid))
        if frame is None or not frame.live or not frame.ports:
            status[int(oid)] = "unresolved"
            continue
        vals: List[int] = []
        for port in frame.ports:
            ce, side = port
            chart = bundle.edge_charts.get(ce)
            if chart is None:
                continue
            raw = OBF.port_side_label(chart, side, state, sys.q)
            val = frame_value_for_port(frame, port, raw)
            if val is not None:
                vals.append(int(val))
        if not vals:
            status[int(oid)] = "unresolved"
        elif len(set(vals)) == 1:
            values[int(oid)] = int(vals[0])
            status[int(oid)] = "coherent"
        else:
            status[int(oid)] = "incoherent"
    return values, status


def section_residual_vector_from_values(
    bundle: BundleObjects,
    values: Dict[int, int],
    edges: Sequence[CompEdge],
) -> Optional[Tuple[int, ...]]:
    bits: List[int] = []
    for ce in edges:
        u, v = int(ce[0]), int(ce[1])
        tr = bundle.transports.get((u, v))
        if tr is None or u not in values or v not in values:
            return None
        su = int(values[u]); sv = int(values[v])
        if su not in tr.mapping:
            return None
        bits.append(0 if int(tr.mapping[su]) == sv else 1)
    return tuple(bits)


def minimal_support_representative(
    bits: Sequence[int],
    basis: CycleBasis,
    max_gauge_flips: int = 4096,
) -> Tuple[Tuple[int, ...], int, bool]:
    """Canonical minimal-support representative modulo Z2 coboundaries.

    The gauge action is r_e -> r_e + g_source + g_target over Z2.  We choose the
    smallest Hamming support representative, breaking ties by integer code.
    For the small controlled observer graphs this is exhaustive.  If the number
    of possible vertex flips exceeds max_gauge_flips, a deterministic truncated
    subset is used and the returned exhaustive flag is false.
    """
    n_nodes = len(basis.nodes)
    n_edges = len(bits)
    if n_edges == 0:
        return tuple(), 0, True
    total = 1 << n_nodes if n_nodes < 62 else max_gauge_flips + 1
    exhaustive = total <= max_gauge_flips
    if exhaustive:
        masks = range(total)
    else:
        # Deterministic low-discrepancy-ish subset: identity, all single flips,
        # then consecutive masks up to the cap.  This keeps large discovered
        # graphs usable without pretending the representative is certified.
        seed_masks = [0] + [1 << i for i in range(min(n_nodes, 60))]
        masks = list(dict.fromkeys(seed_masks + list(range(min(max_gauge_flips, total)))))[:max_gauge_flips]
    node_index = {int(v): i for i, v in enumerate(basis.nodes)}
    best_bits: Optional[Tuple[int, ...]] = None
    best_support: Optional[int] = None
    best_code: Optional[int] = None
    base = tuple(int(b) & 1 for b in bits)
    for mask in masks:
        out = list(base)
        for ei, (u, v) in enumerate(basis.edges):
            gu = (int(mask) >> node_index[int(u)]) & 1
            gv = (int(mask) >> node_index[int(v)]) & 1
            out[ei] ^= (gu ^ gv)
        tb = tuple(out)
        supp = int(sum(tb))
        code = residual_code(tb)
        if best_support is None or (supp, code) < (best_support, int(best_code)):
            best_bits = tb
            best_support = supp
            best_code = code
    return tuple(best_bits or ()), int(best_code or 0), bool(exhaustive)


def analyze_representation_layer(
    name: str,
    counts: Counter,
    transitions: Counter,
    n_bits: int,
    lost_transitions: int = 0,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Analyze schedule quotient/recurrent structure for one residual layer."""
    pattern_rows: List[Dict] = []
    transition_rows: List[Dict] = []
    quotient_rows: List[Dict] = []
    trans_edges: Dict[int, Set[int]] = defaultdict(set)
    for (a, b), c in transitions.items():
        trans_edges[int(a)].add(int(b))
    nodes = set(int(k) for k in counts) | {int(b) for (_a, b) in transitions}
    qmap = deterministic_schedule_quotient(nodes, trans_edges) if nodes else {}
    q_counts: Counter = Counter()
    for code, c in counts.items():
        q_counts[int(qmap.get(int(code), 0))] += int(c)
    q_edges: Dict[int, Set[int]] = defaultdict(set)
    for a, succs in trans_edges.items():
        qa = int(qmap.get(int(a), 0))
        for b in succs:
            q_edges[qa].add(int(qmap.get(int(b), 0)))
    rec_res = recurrent_nodes(nodes, trans_edges) if nodes else set()
    rec_q = recurrent_nodes(set(q_counts), q_edges) if q_counts else set()

    by_src: Dict[int, Counter] = defaultdict(Counter)
    for (a, b), c in transitions.items():
        by_src[int(a)][int(b)] += int(c)
    total_trans = int(sum(transitions.values()) + int(lost_transitions))
    closed_trans = int(sum(transitions.values()))
    best_trans = int(sum(max(cnt.values()) for cnt in by_src.values())) if by_src else 0
    det_acc = float(best_trans / max(1, closed_trans))
    moving = int(sum(c for (a, b), c in transitions.items() if int(a) != int(b)))

    total_count = int(sum(counts.values()))
    for code, c in sorted(counts.items()):
        bits = code_bits(int(code), int(n_bits)) if int(n_bits) > 0 else tuple()
        pattern_rows.append(dict(
            representation=name,
            residual_code=int(code),
            residual_bits="".join(str(b) for b in bits),
            support_size=int(sum(bits)),
            count=int(c),
            frequency=float(c / max(1, total_count)),
            quotient_label=int(qmap.get(int(code), 0)),
            recurrent=int(int(code) in rec_res),
        ))
    for (a, b), c in sorted(transitions.items()):
        transition_rows.append(dict(
            representation=name,
            source_code=int(a), target_code=int(b), count=int(c),
            source_bits=_code_to_bit_string(int(a), int(n_bits)),
            target_bits=_code_to_bit_string(int(b), int(n_bits)),
            source_quotient=int(qmap.get(int(a), 0)),
            target_quotient=int(qmap.get(int(b), 0)),
            moved=int(int(a) != int(b)),
        ))
    for qlab, c in sorted(q_counts.items()):
        succ = sorted(q_edges.get(int(qlab), set()))
        quotient_rows.append(dict(
            representation=name,
            quotient_label=int(qlab),
            count=int(c),
            frequency=float(c / max(1, sum(q_counts.values()))),
            n_successors=int(len(succ)),
            successors=" ".join(map(str, succ)),
            recurrent=int(int(qlab) in rec_q),
        ))
    summary = dict(
        n_patterns=int(len(counts)),
        entropy_bits=float(entropy_counts(counts.values())),
        n_transitions=int(closed_trans),
        lost_transitions=int(lost_transitions),
        closed_transition_fraction=float(closed_trans / max(1, total_trans)),
        transition_determinism_accuracy=float(det_acc),
        moving_transition_fraction=float(moving / max(1, closed_trans)),
        n_recurrent_patterns=int(len(rec_res)),
        n_schedule_quotient_classes=int(len(q_counts)),
        quotient_entropy_bits=float(entropy_counts(q_counts.values())),
        n_recurrent_quotient_classes=int(len(rec_q)),
        nontrivial_recurrent_quotient=bool(len(rec_q) > 1),
    )
    return pattern_rows, transition_rows, quotient_rows, summary


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_section_dynamics(
    sys: OBG.FiniteRelationalSystem,
    q: int,
    rng: np.random.Generator,
    max_channel_inputs: int = 256,
    max_channel_backgrounds: int = 256,
    max_state_samples: int = 4096,
    state_warmup: int = 0,
    schedule_samples: int = 16,
    cycle_max_len: int = 6,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    require_true_transports: bool = True,
    max_local_support: int = 2,
    max_gauge_flips: int = 4096,
) -> Tuple[object, object, object, Dict]:
    """Run the v2 section-obstruction dynamics audit.

    V1 treated every port-level residual pattern as a potential physical
    section obstruction.  V2 separates three layers:

    1. raw port residuals, which may include frame-incoherent states;
    2. coherent section residuals, defined only when every observer has a
       single internally coherent frame value;
    3. cohomological data: cycle syndromes and canonical minimal-support
       representatives modulo Z2 coboundaries.

    Dynamics is only a serious candidate if the coherent/cohomology layers have
    a nontrivial recurrent schedule-invariant quotient.  Raw residual movement
    alone is reported, but not counted as physical motion.
    """
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for section dynamics outputs")
    bundle = build_bundle_objects(
        sys, rng,
        max_channel_inputs=max_channel_inputs,
        max_channel_backgrounds=max_channel_backgrounds,
        max_state_samples=max_state_samples,
        state_warmup=max(0, int(state_warmup)),
        cycle_max_len=cycle_max_len,
        min_frame_ports=min_frame_ports,
        frame_coordinate_mode=frame_coordinate_mode,
    )
    edges = live_transport_edges(bundle, require_true=require_true_transports)
    basis = cycle_basis_for_edges(edges)
    connection_bits = binary_edge_connection_bits(bundle, edges)
    connection_syndrome_bits = syndrome_bits(connection_bits, basis) if connection_bits is not None else tuple()

    states0 = OBF.all_or_sample_states(sys.q, sys.k, max_state_samples, rng)
    if state_warmup > 0:
        warmed = []
        for s0 in states0:
            cur = tuple(s0)
            for _ in range(int(state_warmup)):
                cur = sys.step_parallel(cur)
            warmed.append(cur)
        states0 = warmed
    orders = schedule_orders(sys.k, schedule_samples, rng)

    required_observers = tuple(sorted({x for e in edges for x in e}))

    rep_names = ("raw_port", "coherent_edge", "cycle_syndrome", "minimal_support", "coherence_defect")
    rep_bits = {
        "raw_port": len(edges),
        "coherent_edge": len(edges),
        "cycle_syndrome": basis.beta1,
        "minimal_support": len(edges),
        # Observer-level defect bit: 0 when that observer has a coherent local
        # frame value, 1 when its ports are internally inconsistent/unresolved.
        # This layer lives on observers rather than edges and is defined for
        # every microstate.  It is the natural candidate for localized section
        # dynamics once the observer-frame connection is held fixed.
        "coherence_defect": len(required_observers),
    }
    rep_counts: Dict[str, Counter] = {name: Counter() for name in rep_names}
    rep_transitions: Dict[str, Counter] = {name: Counter() for name in rep_names}
    rep_lost: Dict[str, int] = {name: 0 for name in rep_names}
    rep_current_transition_total: Dict[str, int] = {name: 0 for name in rep_names}

    support_counts: Counter = Counter()
    coherent_support_counts: Counter = Counter()
    min_support_counts: Counter = Counter()
    coherence_defect_support_counts: Counter = Counter()
    coherence_counts: Counter = Counter()
    coherent_status_counts: Counter = Counter()
    minimal_cache: Dict[int, Tuple[Tuple[int, ...], int, bool]] = {}
    minimal_exhaustive = True


    def representations_for_state(state: Sequence[int]) -> Tuple[Dict[str, int], Dict]:
        nonlocal minimal_exhaustive
        reps: Dict[str, int] = {}
        meta: Dict = {}
        raw_bits = residual_vector(sys, state, bundle, edges)
        raw_code = residual_code(raw_bits)
        reps["raw_port"] = raw_code
        meta["raw_bits"] = raw_bits

        values, status = observer_section_values(sys, state, bundle, required_observers)
        n_coh = sum(1 for x in status.values() if x == "coherent")
        n_inc = sum(1 for x in status.values() if x == "incoherent")
        n_unr = sum(1 for x in status.values() if x == "unresolved")
        meta["status_tuple"] = (n_coh, n_inc, n_unr)
        # Observer-level section-coherence defect pattern.  A defect bit is 0
        # only when that observer's incident frame ports define one coherent
        # frame value.  Incoherent and unresolved observers both count as
        # defects because either condition prevents that observer from carrying
        # a local section value.  This representation is defined on every
        # sampled microstate and is the D-stage candidate for localized/moving
        # observer defects after the connection is preserved.
        defect_bits = tuple(0 if status.get(int(oid)) == "coherent" else 1 for oid in required_observers)
        reps["coherence_defect"] = residual_code(defect_bits)
        meta["coherence_defect_bits"] = defect_bits
        all_coherent = bool(required_observers and n_coh == len(required_observers) and n_inc == 0 and n_unr == 0)
        meta["all_coherent"] = all_coherent
        if all_coherent:
            sec_bits = section_residual_vector_from_values(bundle, values, edges)
            if sec_bits is not None:
                sec_code = residual_code(sec_bits)
                reps["coherent_edge"] = sec_code
                syn_bits = syndrome_bits(sec_bits, basis)
                reps["cycle_syndrome"] = residual_code(syn_bits)
                if sec_code not in minimal_cache:
                    minimal_cache[sec_code] = minimal_support_representative(sec_bits, basis, max_gauge_flips=max_gauge_flips)
                min_bits, min_code, ex = minimal_cache[sec_code]
                minimal_exhaustive = bool(minimal_exhaustive and ex)
                reps["minimal_support"] = int(min_code)
                meta["coherent_bits"] = sec_bits
                meta["syndrome_bits"] = syn_bits
                meta["minimal_bits"] = min_bits
        return reps, meta

    # Cache current-state representations so each state is read once before schedules.
    current_cache: Dict[Tuple[int, ...], Tuple[Dict[str, int], Dict]] = {}
    next_cache: Dict[Tuple[int, ...], Tuple[Dict[str, int], Dict]] = {}

    def get_reps(state: Sequence[int]) -> Tuple[Dict[str, int], Dict]:
        key = tuple(int(x) for x in state)
        if key not in next_cache:
            next_cache[key] = representations_for_state(key)
        return next_cache[key]

    for s0 in states0:
        s = tuple(int(x) for x in s0)
        reps, meta = representations_for_state(s)
        current_cache[s] = (reps, meta)
        raw_bits = meta.get("raw_bits", tuple())
        support_counts[int(sum(raw_bits))] += 1
        coherence_counts[observer_frame_coherence(sys, s, bundle)] += 1
        coherent_status_counts[meta.get("status_tuple", (0, 0, 0))] += 1
        for name, code in reps.items():
            rep_counts[name][int(code)] += 1
        if "coherent_bits" in meta:
            coherent_support_counts[int(sum(meta["coherent_bits"]))] += 1
        if "minimal_bits" in meta:
            min_support_counts[int(sum(meta["minimal_bits"]))] += 1
        if "coherence_defect_bits" in meta:
            coherence_defect_support_counts[int(sum(meta["coherence_defect_bits"]))] += 1

        for order in orders:
            nxt = sequential_step(sys, s, order)
            reps_n, _meta_n = get_reps(nxt)
            for name in rep_names:
                if name not in reps:
                    continue
                rep_current_transition_total[name] += 1
                if name in reps_n:
                    rep_transitions[name][(int(reps[name]), int(reps_n[name]))] += 1
                else:
                    rep_lost[name] += 1

    all_pattern_rows: List[Dict] = []
    all_transition_rows: List[Dict] = []
    all_quotient_rows: List[Dict] = []
    layer_summaries: Dict[str, Dict] = {}
    for name in rep_names:
        pr, tr, qr, summ = analyze_representation_layer(
            name,
            rep_counts[name],
            rep_transitions[name],
            rep_bits[name],
            lost_transitions=rep_lost[name],
        )
        all_pattern_rows.extend(pr)
        all_transition_rows.extend(tr)
        all_quotient_rows.extend(qr)
        summ["valid_state_fraction"] = float(sum(rep_counts[name].values()) / max(1, len(states0)))
        summ["current_transition_count"] = int(rep_current_transition_total[name])
        layer_summaries[name] = summ

    raw_s = layer_summaries["raw_port"]
    coh_s = layer_summaries["coherent_edge"]
    syn_s = layer_summaries["cycle_syndrome"]
    min_s = layer_summaries["minimal_support"]
    def_s = layer_summaries["coherence_defect"]

    total_states = int(len(states0))
    raw_total = int(sum(rep_counts["raw_port"].values()))
    coherent_total = int(sum(rep_counts["coherent_edge"].values()))
    active_patterns = sum(c for rc, c in rep_counts["raw_port"].items() if sum(code_bits(rc, len(edges))) > 0)
    localized_patterns = sum(c for rc, c in rep_counts["raw_port"].items() if 0 < sum(code_bits(rc, len(edges))) <= max_local_support)
    coherent_active = sum(c for rc, c in rep_counts["coherent_edge"].items() if sum(code_bits(rc, len(edges))) > 0)
    minimal_active = sum(c for rc, c in rep_counts["minimal_support"].items() if sum(code_bits(rc, len(edges))) > 0)
    coherence_defect_active = sum(c for rc, c in rep_counts["coherence_defect"].items() if sum(code_bits(rc, len(required_observers))) > 0)
    coherence_defect_localized = sum(c for rc, c in rep_counts["coherence_defect"].items() if 0 < sum(code_bits(rc, len(required_observers))) <= max_local_support)

    n_edges = len(edges)
    chord_count = int(bundle.summary.get("global_chord_count", 0))
    nontriv_hol = int(bundle.summary.get("global_nontrivial_holonomy", 0))

    # Conservative verdict: raw representative movement is not enough; the
    # coherent cohomology/minimal-support layers must survive schedule quotienting.
    if n_edges == 0:
        verdict = "NO SECTION-DYNAMICS SIGNAL: no true live frame transports"
    elif chord_count <= 1 and nontriv_hol > 0:
        verdict = "STATIC ONE-LOOP SECTION FRUSTRATION: obstruction exists, but the arena has no independent loop to move into"
    elif coherent_total == 0:
        verdict = "FRAME-INCOHERENT SECTION AUDIT: raw residuals exist, but no fully coherent observer-frame sections were found"
    elif basis.beta1 <= 0:
        verdict = "SECTION RESIDUAL SIGNAL WITHOUT CYCLES: coherent sections exist, but no cycle cohomology is available"
    elif def_s["nontrivial_recurrent_quotient"] and def_s["closed_transition_fraction"] >= 0.999 and def_s["moving_transition_fraction"] > 0:
        verdict = "COHERENCE-DEFECT DYNAMICS CANDIDATE: observer-level coherence defects have nontrivial recurrent schedule-invariant dynamics"
    elif min_s["nontrivial_recurrent_quotient"] and min_s["closed_transition_fraction"] >= 0.999 and min_s["moving_transition_fraction"] > 0:
        verdict = "SECTION-OBSTRUCTION COHOMOLOGY DYNAMICS CANDIDATE: coherent minimal-support residuals have nontrivial recurrent schedule-invariant dynamics"
    elif min_s["closed_transition_fraction"] < 0.999:
        verdict = "COHERENT SECTION SUBSPACE NOT CLOSED: schedules leave the coherent-section sector, so no autonomous section dynamics is established"
    elif syn_s["n_patterns"] <= 1 and min_s["n_patterns"] <= 1:
        verdict = "STATIC COHOMOLOGY SECTION OBSTRUCTION: raw representatives may move, but coherent syndrome/minimal-support class is fixed"
    elif raw_s["nontrivial_recurrent_quotient"] and not min_s["nontrivial_recurrent_quotient"]:
        verdict = "REPRESENTATIVE-LEVEL SECTION SIGNAL ONLY: raw residual quotient is nontrivial but cohomology/minimal-support dynamics collapses"
    elif min_s["n_schedule_quotient_classes"] <= 1:
        verdict = "SCHEDULE-GAUGE COLLAPSE OF SECTION DYNAMICS: coherent residuals vary, but deterministic quotient is trivial"
    else:
        verdict = "SECTION-OBSTRUCTION V2 QUOTIENT SIGNAL: coherent residual structure exists, motion not established"

    summary = dict(
        verdict=verdict,
        audit_version="section_dynamics_v2_coherent_cohomology",
        q=int(q),
        n_vertices=int(sys.k),
        n_observers=int(len(bundle.comps)),
        n_required_section_observers=int(len(required_observers)),
        n_live_transport_edges=int(n_edges),
        observer_cycles=int(len(bundle.cycles)),
        cycle_basis_size=int(basis.beta1),
        cycle_basis_chords=[f"{a}->{b}" for a, b in basis.chords],
        cycle_basis_rows=[" ".join(str(int(i)) for i in row) for row in basis.basis],
        connection_edge_bits="" if connection_bits is None else "".join(str(int(b)) for b in connection_bits),
        connection_holonomy_bits="".join(str(int(b)) for b in connection_syndrome_bits),
        connection_holonomy_weight=int(sum(connection_syndrome_bits)),
        minimal_representative_exhaustive=bool(minimal_exhaustive),
        max_gauge_flips=int(max_gauge_flips),
        global_chord_count=int(bundle.summary.get("global_chord_count", 0)),
        global_valid_holonomy=int(bundle.summary.get("global_valid_holonomy", 0)),
        global_nontrivial_holonomy=int(bundle.summary.get("global_nontrivial_holonomy", 0)),
        global_generated_group=str(bundle.summary.get("global_generated_group", "none")),
        bundle_summary=bundle.summary,
        n_sampled_states=total_states,
        n_schedules=int(len(orders)),
        # Backward-compatible raw residual fields.
        n_residual_patterns=int(raw_s["n_patterns"]),
        residual_entropy_bits=float(raw_s["entropy_bits"]),
        mean_support_size=float(sum(k * v for k, v in support_counts.items()) / max(1, sum(support_counts.values()))),
        active_residual_fraction=float(active_patterns / max(1, raw_total)),
        localized_residual_fraction=float(localized_patterns / max(1, raw_total)),
        transition_determinism_accuracy=float(raw_s["transition_determinism_accuracy"]),
        moving_transition_fraction=float(raw_s["moving_transition_fraction"]),
        n_recurrent_residual_patterns=int(raw_s["n_recurrent_patterns"]),
        n_schedule_quotient_classes=int(raw_s["n_schedule_quotient_classes"]),
        quotient_entropy_bits=float(raw_s["quotient_entropy_bits"]),
        n_recurrent_quotient_classes=int(raw_s["n_recurrent_quotient_classes"]),
        nontrivial_recurrent_quotient=bool(raw_s["nontrivial_recurrent_quotient"]),
        # V2 coherent-section fields.
        coherent_state_count=int(coherent_total),
        coherent_state_fraction=float(coherent_total / max(1, total_states)),
        frame_incoherent_or_unresolved_fraction=float(1.0 - coherent_total / max(1, total_states)),
        coherent_residual_patterns=int(coh_s["n_patterns"]),
        coherent_residual_entropy_bits=float(coh_s["entropy_bits"]),
        coherent_active_residual_fraction=float(coherent_active / max(1, coherent_total)),
        coherent_transition_closed_fraction=float(coh_s["closed_transition_fraction"]),
        coherent_transition_determinism_accuracy=float(coh_s["transition_determinism_accuracy"]),
        coherent_moving_transition_fraction=float(coh_s["moving_transition_fraction"]),
        coherent_schedule_quotient_classes=int(coh_s["n_schedule_quotient_classes"]),
        coherent_recurrent_quotient_classes=int(coh_s["n_recurrent_quotient_classes"]),
        coherent_nontrivial_recurrent_quotient=bool(coh_s["nontrivial_recurrent_quotient"]),
        # V2 cohomology/syndrome fields.
        cycle_syndrome_patterns=int(syn_s["n_patterns"]),
        cycle_syndrome_entropy_bits=float(syn_s["entropy_bits"]),
        cycle_syndrome_transition_closed_fraction=float(syn_s["closed_transition_fraction"]),
        cycle_syndrome_moving_transition_fraction=float(syn_s["moving_transition_fraction"]),
        cycle_syndrome_schedule_quotient_classes=int(syn_s["n_schedule_quotient_classes"]),
        cycle_syndrome_recurrent_quotient_classes=int(syn_s["n_recurrent_quotient_classes"]),
        cycle_syndrome_nontrivial_recurrent_quotient=bool(syn_s["nontrivial_recurrent_quotient"]),
        minimal_support_patterns=int(min_s["n_patterns"]),
        minimal_support_entropy_bits=float(min_s["entropy_bits"]),
        minimal_support_mean_size=float(sum(k * v for k, v in min_support_counts.items()) / max(1, sum(min_support_counts.values()))),
        minimal_support_active_fraction=float(minimal_active / max(1, coherent_total)),
        minimal_support_transition_closed_fraction=float(min_s["closed_transition_fraction"]),
        minimal_support_moving_transition_fraction=float(min_s["moving_transition_fraction"]),
        minimal_support_schedule_quotient_classes=int(min_s["n_schedule_quotient_classes"]),
        minimal_support_recurrent_quotient_classes=int(min_s["n_recurrent_quotient_classes"]),
        minimal_support_nontrivial_recurrent_quotient=bool(min_s["nontrivial_recurrent_quotient"]),
        # V2b observer-level coherence-defect fields.  These are defined for
        # all states and live on observer nodes rather than connection edges.
        coherence_defect_patterns=int(def_s["n_patterns"]),
        coherence_defect_entropy_bits=float(def_s["entropy_bits"]),
        coherence_defect_active_fraction=float(coherence_defect_active / max(1, raw_total)),
        coherence_defect_localized_fraction=float(coherence_defect_localized / max(1, raw_total)),
        coherence_defect_transition_closed_fraction=float(def_s["closed_transition_fraction"]),
        coherence_defect_transition_determinism_accuracy=float(def_s["transition_determinism_accuracy"]),
        coherence_defect_moving_transition_fraction=float(def_s["moving_transition_fraction"]),
        coherence_defect_schedule_quotient_classes=int(def_s["n_schedule_quotient_classes"]),
        coherence_defect_recurrent_quotient_classes=int(def_s["n_recurrent_quotient_classes"]),
        coherence_defect_nontrivial_recurrent_quotient=bool(def_s["nontrivial_recurrent_quotient"]),
        max_local_support=int(max_local_support),
        support_size_counts={str(int(k)): int(v) for k, v in sorted(support_counts.items())},
        coherent_support_size_counts={str(int(k)): int(v) for k, v in sorted(coherent_support_counts.items())},
        minimal_support_size_counts={str(int(k)): int(v) for k, v in sorted(min_support_counts.items())},
        coherence_defect_support_size_counts={str(int(k)): int(v) for k, v in sorted(coherence_defect_support_counts.items())},
        frame_coherence_counts={str(k): int(v) for k, v in coherence_counts.items()},
        section_status_counts={str(k): int(v) for k, v in coherent_status_counts.items()},
        live_transport_edges=[f"{a}->{b}" for a, b in edges],
        layer_summaries=layer_summaries,
    )
    return pd.DataFrame(all_pattern_rows), pd.DataFrame(all_transition_rows), pd.DataFrame(all_quotient_rows), summary


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------
def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def write_outputs(pattern_df, transition_df, quotient_df, summary: Dict, out: str) -> None:
    pattern_df.to_csv(out, index=False)
    transition_df.to_csv(derived_path(out, "_transitions"), index=False)
    quotient_df.to_csv(derived_path(out, "_quotient"), index=False)
    with open(derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def make_plot(pattern_df, transition_df, quotient_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle(summary.get("verdict", "section obstruction dynamics"), fontsize=10)
    raw_df = pattern_df
    min_df = pattern_df
    if len(pattern_df) and "representation" in pattern_df.columns:
        raw_df = pattern_df[pattern_df["representation"] == "raw_port"]
        min_df = pattern_df[pattern_df["representation"] == "minimal_support"]
    if len(raw_df):
        axes[0].hist(raw_df["support_size"].astype(float), bins=range(0, int(raw_df["support_size"].max()) + 3))
    axes[0].set_title("raw residual support")
    axes[0].set_xlabel("active edges")
    axes[0].set_ylabel("patterns")
    if len(transition_df):
        tdf = transition_df
        if "representation" in tdf.columns:
            tdf = tdf[tdf["representation"] == "minimal_support"]
            if len(tdf) == 0:
                tdf = transition_df[transition_df["representation"] == "raw_port"]
        axes[1].hist(tdf["moved"].astype(float), bins=[-0.5, 0.5, 1.5])
    axes[1].set_title("minimal/coherent moved?")
    axes[1].set_xlabel("0/1")
    if len(min_df):
        axes[2].hist(min_df["support_size"].astype(float), bins=range(0, int(min_df["support_size"].max()) + 3))
    axes[2].set_title("minimal support reps")
    axes[2].set_xlabel("active edges")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Observer-frame section obstruction dynamics audit")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--winners", default="", help="optional winners pickle; if supplied, graph-mode is ignored")
    p.add_argument("--winner-index", type=int, default=0)
    p.add_argument("--graph-mode", choices=[
        "frame_twist_c2", "frame_flat_c2", "frame_random_diamond",
        "frame_theta_flat_c2", "frame_theta_twist_c2", "frame_random_theta",
        "frame_double_flat_c2", "frame_double_twist_c2", "frame_random_double_diamond",
        "componented", "er"], default="frame_theta_twist_c2")
    p.add_argument("--vertices", type=int, default=18)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.16)
    p.add_argument("--inter-prob", type=float, default=0.35)
    p.add_argument("--extra-intra-prob", type=float, default=0.25)
    p.add_argument("--max-pred", type=int, default=4)
    p.add_argument("--max-channel-inputs", type=int, default=256)
    p.add_argument("--max-channel-backgrounds", type=int, default=256)
    p.add_argument("--max-state-samples", type=int, default=4096)
    p.add_argument("--state-warmup", type=int, default=0)
    p.add_argument("--schedule-samples", type=int, default=16)
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--min-frame-ports", type=int, default=2)
    p.add_argument("--frame-coordinate-mode", choices=["local_charts", "common_factor"], default="local_charts")
    p.add_argument("--max-local-support", type=int, default=2)
    p.add_argument("--max-gauge-flips", type=int, default=4096,
                   help="maximum vertex-gauge flips for minimal-support cohomology representative")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/observer_section_dynamics.csv")
    p.add_argument("--plot", default="example_results/fig_observer_section_dynamics.png")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rng = np.random.default_rng(int(args.seed))
    sys = make_system_from_args(args, rng)
    pattern_df, transition_df, quotient_df, summary = run_section_dynamics(
        sys=sys,
        q=int(args.q),
        rng=rng,
        max_channel_inputs=int(args.max_channel_inputs),
        max_channel_backgrounds=int(args.max_channel_backgrounds),
        max_state_samples=int(args.max_state_samples),
        state_warmup=int(args.state_warmup),
        schedule_samples=int(args.schedule_samples),
        cycle_max_len=int(args.cycle_max_len),
        min_frame_ports=int(args.min_frame_ports),
        frame_coordinate_mode=str(args.frame_coordinate_mode),
        max_local_support=int(args.max_local_support),
        max_gauge_flips=int(args.max_gauge_flips),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_outputs(pattern_df, transition_df, quotient_df, summary, args.out)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plot(pattern_df, transition_df, quotient_df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {derived_path(args.out, '_transitions')}")
    print(f"wrote {derived_path(args.out, '_quotient')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
