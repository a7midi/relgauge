"""
observerhiddenmemory.py

Hidden observer-memory audit for observer-frame connections.

Motivation
----------
The section/coherence-defect audits treat each observer mostly by its visible
frame value or by whether its boundary ports are coherent.  The latest gated
D-stage runs preserve the theta observer-frame connection, but visible residual
and coherence-defect dynamics remain schedule-nondeterministic.

This module tests a different logical possibility:

    An SCC observer has internal microstate hidden from other observers.  The
    visible boundary/frame output can be schedule-ambiguous by itself, but the
    ambiguity may be absorbed by a minimal quotient of the observer-internal
    state.

Nothing physical is inserted.  The hidden variables are the existing internal
states of the SCC observers.  The module asks whether conditioning on a derived
hidden-memory quotient makes visible observer-frame dynamics schedule-consistent.

Core test
---------
For every sampled microscopic state x:

    V(x) = tuple of observer-visible frame/coherence outputs.
    H(x) = tuple of internal SCC microstate codes for the same observers.
    J(x) = (V(x), H(x)).

For sampled update schedules sigma, compute x' = T_sigma(x).  The audit compares:

    V  -> V'                 visible-only dynamics
    J  -> V'                 hidden-conditioned visible dynamics
    J  -> J'                 full hidden dynamics

It also computes a color-preserving deterministic quotient of J-states, where
colors are visible outputs V.  This quotient may identify hidden successor states
created by schedule ambiguity, but it is not allowed to identify states with
different visible outputs.  Therefore, if the quotient becomes deterministic, the
hidden memory is absorbing schedule ambiguity without erasing visible distinctions.

This is an audit, not a matter/particle selector.  It does not reward motion,
nontrivial recurrent quotients, C2, flux, or particles; those are reported only
post hoc.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import observerboundarygeometry as OBG
    from . import observerframebundle as OBF
    from . import observersectiondynamics as OSD
    from . import blindobserverconnection as BOC
except Exception:  # pragma: no cover
    import observerboundarygeometry as OBG  # type: ignore
    import observerframebundle as OBF  # type: ignore
    import observersectiondynamics as OSD  # type: ignore
    import blindobserverconnection as BOC  # type: ignore

CompEdge = Tuple[int, int]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def entropy_counts(counts: Iterable[int]) -> float:
    arr = np.asarray([float(c) for c in counts if c > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log2(p)).sum())


def encode_tuple_base(vals: Sequence[int], base: int) -> int:
    code = 0
    for x in vals:
        code = code * int(base) + int(x)
    return int(code)


def all_or_sample_states(q: int, k: int, max_states: int, rng: np.random.Generator) -> List[Tuple[int, ...]]:
    total = int(q) ** int(k)
    if max_states <= 0 or total <= int(max_states):
        return [tuple(int(x) for x in s) for s in itertools.product(range(q), repeat=k)]
    return [tuple(int(x) for x in rng.integers(0, q, size=k)) for _ in range(int(max_states))]


class UnionFind:
    def __init__(self, nodes: Iterable[int]):
        vals = sorted(set(int(x) for x in nodes))
        self.parent = {x: x for x in vals}
        self.rank = {x: 0 for x in vals}

    def find(self, x: int) -> int:
        x = int(x)
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def recurrent_nodes(nodes: Set[int], edges: Dict[int, Set[int]]) -> Set[int]:
    idx = 0
    stack: List[int] = []
    onstack: Set[int] = set()
    indices: Dict[int, int] = {}
    low: Dict[int, int] = {}
    out: Set[int] = set()

    def strong(v: int) -> None:
        nonlocal idx
        indices[v] = idx
        low[v] = idx
        idx += 1
        stack.append(v)
        onstack.add(v)
        for w in edges.get(v, set()):
            if w not in indices:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or (len(comp) == 1 and comp[0] in edges.get(comp[0], set())):
                out.update(comp)

    for v in sorted(nodes):
        if v not in indices:
            strong(v)
    return out




def _tarjan_components(nodes: Set[int], edges: Dict[int, Set[int]]) -> List[List[int]]:
    """Strongly connected components of a finite directed relation."""
    node_set = set(int(n) for n in nodes)
    for a, succs in edges.items():
        node_set.add(int(a))
        node_set.update(int(s) for s in succs)
    idx = 0
    stack: List[int] = []
    onstack: Set[int] = set()
    indices: Dict[int, int] = {}
    low: Dict[int, int] = {}
    comps: List[List[int]] = []

    def strong(v: int) -> None:
        nonlocal idx
        indices[v] = idx
        low[v] = idx
        idx += 1
        stack.append(v)
        onstack.add(v)
        for w in edges.get(int(v), set()):
            w = int(w)
            if w not in indices:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(int(w))
                if w == v:
                    break
            comps.append(sorted(comp))

    for v in sorted(node_set):
        if v not in indices:
            strong(int(v))
    return comps


def recurrent_structure_summary(nodes: Set[int], edges: Dict[int, Set[int]], active_counts: Optional[Counter] = None) -> Dict:
    """Classify recurrent components and periods of a quotient transition graph.

    ``recurrent_classes`` alone is ambiguous: one recurrent component can be a
    fixed point or a genuine deterministic cycle.  This helper reports both the
    number of recurrent components and their sizes/periods.  When each node in a
    recurrent component has exactly one successor inside that component, its
    period is the component size; otherwise the component is marked as
    nondeterministic and only its SCC size is reported.
    """
    node_set = set(int(n) for n in nodes)
    for a, succs in edges.items():
        node_set.add(int(a))
        node_set.update(int(s) for s in succs)
    if not node_set:
        return dict(
            recurrent_component_count=0,
            recurrent_state_count=0,
            recurrent_component_sizes=[],
            recurrent_cycle_lengths=[],
            max_recurrent_period=0,
            fixed_point_count=0,
            nontrivial_periodic_recurrence=False,
            recurrent_relation_deterministic=True,
        )
    active_nodes = set(int(n) for n in (active_counts.keys() if active_counts is not None else node_set))
    comps = _tarjan_components(node_set, edges)
    recurrent_comps: List[List[int]] = []
    for comp in comps:
        cset = set(comp)
        has_cycle = len(comp) > 1 or any(int(v) in edges.get(int(v), set()) for v in comp)
        # recurrent components are closed SCCs: no outgoing edge leaves the SCC.
        closed = all(set(int(s) for s in edges.get(int(v), set())).issubset(cset) for v in comp)
        if has_cycle and closed and any(v in active_nodes for v in comp):
            recurrent_comps.append(sorted([v for v in comp if v in active_nodes or v in cset]))

    sizes: List[int] = []
    cycle_lengths: List[int] = []
    fixed_points = 0
    deterministic = True
    for comp in recurrent_comps:
        cset = set(comp)
        sizes.append(len(comp))
        succ_inside = {v: sorted(set(int(s) for s in edges.get(int(v), set())) & cset) for v in comp}
        is_functional = all(len(succ_inside[v]) == 1 for v in comp)
        deterministic = deterministic and is_functional
        if is_functional:
            if len(comp) == 1 and succ_inside[comp[0]] == [comp[0]]:
                fixed_points += 1
                cycle_lengths.append(1)
            elif len(comp) > 1:
                # In a functional SCC every node lies on one directed cycle; SCC
                # size equals period.
                cycle_lengths.append(len(comp))
            else:
                cycle_lengths.append(0)
        else:
            cycle_lengths.append(0)
    return dict(
        recurrent_component_count=int(len(recurrent_comps)),
        recurrent_state_count=int(sum(sizes)),
        recurrent_component_sizes=[int(x) for x in sizes],
        recurrent_cycle_lengths=[int(x) for x in cycle_lengths],
        max_recurrent_period=int(max(cycle_lengths) if cycle_lengths else 0),
        fixed_point_count=int(fixed_points),
        nontrivial_periodic_recurrence=bool(any(int(x) > 1 for x in cycle_lengths)),
        recurrent_relation_deterministic=bool(deterministic),
    )

def transition_accuracy(transitions: Counter) -> Tuple[float, float, int, int]:
    """Return deterministic accuracy, moving fraction, source count, transition count.

    Keys may be integers or structured visible symbols; this helper intentionally
    treats them as hashable labels rather than numeric codes.
    """
    by_src: Dict[object, Counter] = defaultdict(Counter)
    total = 0
    moving = 0
    for (a, b), c in transitions.items():
        by_src[a][b] += int(c)
        total += int(c)
        if a != b:
            moving += int(c)
    best = sum(max(cn.values()) for cn in by_src.values()) if by_src else 0
    return float(best / max(1, total)), float(moving / max(1, total)), int(len(by_src)), int(total)


def transition_edges(transitions: Counter) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for (a, b), _c in transitions.items():
        out[int(a)].add(int(b))
    return out


def quotient_transition_summary(
    q_counts: Counter,
    q_edges: Dict[int, Set[int]],
    q_colors: Optional[Dict[int, int]] = None,
) -> Dict:
    """Summarize a finite quotient transition relation.

    The older D-stage summaries only reported ``recurrent_quotient_classes``.
    That is insufficient: one recurrent component can be a fixed point or a
    nontrivial deterministic cycle.  This v3 audit therefore also reports
    recurrent component sizes, cycle lengths, max period, and fixed-point count.
    """
    q_nodes = set(int(q) for q in q_counts) | set(int(q) for q in q_edges)
    for succs in q_edges.values():
        q_nodes.update(int(s) for s in succs)
    rec_all = recurrent_nodes(q_nodes, q_edges) if q_nodes else set()
    active_nodes = set(int(q) for q in q_counts)
    rec_q = rec_all & active_nodes
    source_blocks = sum(1 for q in active_nodes if q in q_edges)
    deterministic_blocks = sum(1 for q in active_nodes if len(q_edges.get(int(q), set())) <= 1)
    deterministic_sources = sum(1 for q in active_nodes if q in q_edges and len(q_edges.get(int(q), set())) <= 1)
    moving_edges = 0
    total_edges = 0
    visible_moving_edges = 0
    for q, succs in q_edges.items():
        for s in succs:
            total_edges += 1
            if int(q) != int(s):
                moving_edges += 1
            if q_colors is not None and q_colors.get(int(q), -1) != q_colors.get(int(s), -1):
                visible_moving_edges += 1
    rec_struct = recurrent_structure_summary(q_nodes, q_edges, q_counts)
    out = dict(
        quotient_classes=int(len(q_counts)),
        quotient_entropy_bits=float(entropy_counts(q_counts.values())),
        recurrent_quotient_classes=int(len(rec_q)),
        nontrivial_recurrent_quotient=bool(len(rec_q) > 1),
        deterministic_block_fraction=float(deterministic_blocks / max(1, len(q_nodes))),
        source_block_deterministic_fraction=float(deterministic_sources / max(1, source_blocks)),
        quotient_transition_edges=int(total_edges),
        quotient_moving_edge_fraction=float(moving_edges / max(1, total_edges)),
        quotient_visible_moving_edge_fraction=float(visible_moving_edges / max(1, total_edges)) if q_colors is not None else 0.0,
    )
    out.update(rec_struct)
    return out


def deterministic_schedule_quotient(
    nodes: Set[int],
    edges: Dict[int, Set[int]],
    counts: Counter,
    colors: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, int], Dict, List[Dict]]:
    """Finest quotient forced by schedule-nondeterministic successors.

    Starting from the discrete partition, whenever a block has multiple successor
    blocks, those successor blocks are identified.  Iterating this rule gives
    the finest quotient in which every source block has at most one successor
    block.  Unlike ``color_preserving_deterministic_quotient``, this routine is
    allowed to merge different visible colors; this is the visible-quotienting
    correction used to test whether raw visible distinctions were schedule-gauge.
    """
    all_nodes = set(int(n) for n in nodes)
    for a, succs in edges.items():
        all_nodes.add(int(a))
        all_nodes.update(int(s) for s in succs)
    if not all_nodes:
        return {}, dict(), []
    uf = UnionFind(all_nodes)
    changed = True
    while changed:
        changed = False
        blocks: Dict[int, List[int]] = defaultdict(list)
        for n in sorted(all_nodes):
            blocks[uf.find(n)].append(int(n))
        for _r, members in blocks.items():
            succ_roots = sorted({uf.find(s) for m in members for s in edges.get(int(m), set())})
            if len(succ_roots) > 1:
                first = int(succ_roots[0])
                for other in succ_roots[1:]:
                    if uf.union(first, int(other)):
                        changed = True
    roots = sorted({uf.find(n) for n in all_nodes})
    root_to_q = {r: i for i, r in enumerate(roots)}
    qmap = {int(n): int(root_to_q[uf.find(n)]) for n in all_nodes}

    q_counts: Counter = Counter()
    q_color_sets: Dict[int, Set[int]] = defaultdict(set)
    q_color_major: Dict[int, int] = {}
    for n, c in counts.items():
        if int(n) not in qmap:
            continue
        q = int(qmap[int(n)])
        q_counts[q] += int(c)
        if colors is not None:
            q_color_sets[q].add(int(colors.get(int(n), -1)))
    if colors is not None:
        for q, cs in q_color_sets.items():
            if not cs:
                q_color_major[q] = -1
                continue
            # Majority visible color within this quotient block, weighted by counts.
            cc = Counter()
            for n, c in counts.items():
                if int(n) in qmap and qmap[int(n)] == q:
                    cc[int(colors.get(int(n), -1))] += int(c)
            q_color_major[q] = int(cc.most_common(1)[0][0]) if cc else int(min(cs))

    q_edges: Dict[int, Set[int]] = defaultdict(set)
    for a, succs in edges.items():
        if int(a) not in qmap:
            continue
        qa = int(qmap[int(a)])
        for b in succs:
            if int(b) in qmap:
                q_edges[qa].add(int(qmap[int(b)]))

    summary = quotient_transition_summary(q_counts, q_edges, q_color_major if colors is not None else None)
    if colors is not None:
        mixed = [q for q, cs in q_color_sets.items() if len(cs) > 1]
        summary.update(
            visible_mixed_blocks=int(len(mixed)),
            visible_mixed_block_fraction=float(len(mixed) / max(1, len(q_counts))),
            mean_visible_colors_per_block=float(np.mean([len(cs) for cs in q_color_sets.values()])) if q_color_sets else 0.0,
            max_visible_colors_per_block=int(max([len(cs) for cs in q_color_sets.values()] or [0])),
        )
    rows: List[Dict] = []
    total = int(sum(q_counts.values()))
    for q, c in sorted(q_counts.items()):
        succ = sorted(q_edges.get(int(q), set()))
        rows.append(dict(
            quotient_label=int(q),
            count=int(c),
            frequency=float(c / max(1, total)),
            n_successors=int(len(succ)),
            successors=" ".join(map(str, succ)),
            recurrent=int(int(q) in recurrent_nodes(set(q_counts), q_edges) if q_counts else 0),
            visible_color_major=int(q_color_major.get(int(q), -1)) if colors is not None else -1,
            n_visible_colors=int(len(q_color_sets.get(int(q), set()))) if colors is not None else 0,
            visible_colors=" ".join(map(str, sorted(q_color_sets.get(int(q), set())))) if colors is not None else "",
        ))
    return qmap, summary, rows


def visible_partition_induced_by_joint_quotient(
    joint_qmap: Dict[int, int],
    id_to_joint: Dict[int, Tuple[int, int]],
    visible_counts: Counter,
    visible_transitions: Counter,
) -> Tuple[Dict[int, int], Dict, List[Dict]]:
    """Coarsen visible states that the joint schedule quotient puts together."""
    visible_nodes = set(int(v) for v in visible_counts)
    uf = UnionFind(visible_nodes)
    by_q: Dict[int, Set[int]] = defaultdict(set)
    for jid, qlab in joint_qmap.items():
        if int(jid) in id_to_joint:
            vid, _hid = id_to_joint[int(jid)]
            by_q[int(qlab)].add(int(vid))
    for vids in by_q.values():
        vals = sorted(vids)
        if len(vals) > 1:
            first = vals[0]
            for v in vals[1:]:
                uf.union(first, v)
    roots = sorted({uf.find(v) for v in visible_nodes})
    root_to_q = {r: i for i, r in enumerate(roots)}
    vqmap = {int(v): int(root_to_q[uf.find(v)]) for v in visible_nodes}
    q_counts: Counter = Counter()
    for v, c in visible_counts.items():
        q_counts[int(vqmap[int(v)])] += int(c)
    q_edges: Dict[int, Set[int]] = defaultdict(set)
    q_trans: Counter = Counter()
    for (a, b), c in visible_transitions.items():
        if int(a) in vqmap and int(b) in vqmap:
            qa, qb = int(vqmap[int(a)]), int(vqmap[int(b)])
            q_edges[qa].add(qb)
            q_trans[(qa, qb)] += int(c)
    acc, moving, _src, _total = transition_accuracy(q_trans)
    summary = quotient_transition_summary(q_counts, q_edges)
    summary.update(
        transition_determinism_accuracy=float(acc),
        moving_transition_fraction=float(moving),
    )
    rows = []
    total = int(sum(q_counts.values()))
    members: Dict[int, List[int]] = defaultdict(list)
    for v, q in vqmap.items():
        members[int(q)].append(int(v))
    for q, c in sorted(q_counts.items()):
        rows.append(dict(
            quotient_label=int(q),
            count=int(c),
            frequency=float(c / max(1, total)),
            n_visible_states=int(len(members.get(int(q), []))),
            visible_states=" ".join(map(str, sorted(members.get(int(q), [])))),
            n_successors=int(len(q_edges.get(int(q), set()))),
            successors=" ".join(map(str, sorted(q_edges.get(int(q), set())))),
        ))
    return vqmap, summary, rows


# ---------------------------------------------------------------------------
# Connection-preservation gate, copied locally to avoid depending on D-stage selector
# ---------------------------------------------------------------------------
def default_connection_gate_targets(graph_mode: str) -> Dict[str, int]:
    gm = str(graph_mode).lower()
    if "theta" in gm:
        return dict(live_edge_quotients=6, true_frames=5, true_transports=6,
                    cycle_basis=2, global_chords=2, global_holonomy=2)
    if "double" in gm:
        return dict(live_edge_quotients=8, true_frames=7, true_transports=8,
                    cycle_basis=2, global_chords=2, global_holonomy=2)
    if "diamond" in gm or "flat_c2" in gm or "twist_c2" in gm:
        return dict(live_edge_quotients=4, true_frames=4, true_transports=4,
                    cycle_basis=1, global_chords=1, global_holonomy=1)
    return dict(live_edge_quotients=0, true_frames=0, true_transports=0,
                cycle_basis=0, global_chords=0, global_holonomy=0)


def _ratio(actual: float, target: float) -> float:
    if float(target) <= 0:
        return 1.0
    return max(0.0, min(1.0, float(actual) / float(target)))


def connection_gate_status(
    section_summary: Dict,
    gate_min_live_edge_quotients: int = 0,
    gate_min_true_frames: int = 0,
    gate_min_true_transports: int = 0,
    gate_min_cycle_basis: int = 0,
    gate_min_global_chords: int = 0,
    gate_min_global_holonomy: int = 0,
    require_no_single_port_leakage: bool = True,
) -> Tuple[bool, float, Dict]:
    bundle = dict(section_summary.get("bundle_summary", {}) or {})
    actual = dict(
        live_edge_quotients=int(bundle.get("n_live_edge_quotients", 0)),
        true_frames=int(bundle.get("n_true_live_frames", 0)),
        true_transports=int(bundle.get("n_true_live_frame_transports", 0)),
        cycle_basis=int(section_summary.get("cycle_basis_size", 0)),
        global_chords=int(section_summary.get("global_chord_count", bundle.get("global_chord_count", 0))),
        global_holonomy=int(section_summary.get("global_valid_holonomy", bundle.get("global_valid_holonomy", 0))),
        full_boundary_violation=int(bundle.get("full_boundary_violation", section_summary.get("full_boundary_violation", 0))),
        single_port_label_fraction=float(bundle.get("single_port_label_frame_fraction", 0.0)),
    )
    target = dict(
        live_edge_quotients=int(gate_min_live_edge_quotients),
        true_frames=int(gate_min_true_frames),
        true_transports=int(gate_min_true_transports),
        cycle_basis=int(gate_min_cycle_basis),
        global_chords=int(gate_min_global_chords),
        global_holonomy=int(gate_min_global_holonomy),
    )
    ratios = {k: _ratio(actual[k], v) for k, v in target.items() if int(v) > 0}
    if require_no_single_port_leakage:
        ratios["no_single_port_leakage"] = 1.0 if actual["single_port_label_fraction"] <= 1e-12 else 0.0
    ratios["no_full_boundary_violation"] = 1.0 if actual["full_boundary_violation"] == 0 else 0.0
    progress = float(np.mean(list(ratios.values()))) if ratios else 1.0
    passed = bool(progress >= 0.999999)
    return passed, progress, dict(actual=actual, target=target, ratios=ratios, progress=progress, passed=passed)


# ---------------------------------------------------------------------------
# Visible / hidden readouts
# ---------------------------------------------------------------------------
def internal_code(sys: OBG.FiniteRelationalSystem, state: Sequence[int], nodes: Sequence[int]) -> int:
    return encode_tuple_base([int(state[int(v)]) for v in nodes], int(sys.q))


def observer_visible_symbol(
    sys: OBG.FiniteRelationalSystem,
    state: Sequence[int],
    bundle: OSD.BundleObjects,
    observer_id: int,
) -> Tuple[str, int]:
    """Visible observer-frame output: coherent value, incoherent, unresolved.

    This is deliberately coarser than the full SCC state.  A coherent observer
    exposes its frame value.  An incoherent observer exposes that it has no single
    frame value.  An unresolved observer exposes missing/no live port data.
    """
    frame = bundle.frames.get(int(observer_id))
    if frame is None or not frame.live or not frame.ports:
        return ("unresolved", -1)
    vals: List[int] = []
    for port in frame.ports:
        ce, side = port
        chart = bundle.edge_charts.get(ce)
        if chart is None:
            continue
        raw = OBF.port_side_label(chart, side, state, sys.q)
        val = OSD.frame_value_for_port(frame, port, raw)
        if val is not None:
            vals.append(int(val))
    if not vals:
        return ("unresolved", -1)
    if len(set(vals)) == 1:
        return ("coherent", int(vals[0]))
    return ("incoherent", -1)


def state_visible_hidden(
    sys: OBG.FiniteRelationalSystem,
    state: Sequence[int],
    bundle: OSD.BundleObjects,
    observers: Sequence[int],
) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[int, ...]]:
    visible = tuple(observer_visible_symbol(sys, state, bundle, oid) for oid in observers)
    hidden = tuple(internal_code(sys, state, bundle.comps[int(oid)]) for oid in observers)
    return visible, hidden


@dataclass
class Encoders:
    visible_to_id: Dict[Tuple[Tuple[str, int], ...], int]
    id_to_visible: Dict[int, Tuple[Tuple[str, int], ...]]
    hidden_to_id: Dict[Tuple[int, ...], int]
    id_to_hidden: Dict[int, Tuple[int, ...]]
    joint_to_id: Dict[Tuple[int, int], int]
    id_to_joint: Dict[int, Tuple[int, int]]

    @classmethod
    def empty(cls) -> "Encoders":
        return cls({}, {}, {}, {}, {}, {})

    def visible_id(self, visible: Tuple[Tuple[str, int], ...]) -> int:
        if visible not in self.visible_to_id:
            i = len(self.visible_to_id)
            self.visible_to_id[visible] = i
            self.id_to_visible[i] = visible
        return int(self.visible_to_id[visible])

    def hidden_id(self, hidden: Tuple[int, ...]) -> int:
        if hidden not in self.hidden_to_id:
            i = len(self.hidden_to_id)
            self.hidden_to_id[hidden] = i
            self.id_to_hidden[i] = hidden
        return int(self.hidden_to_id[hidden])

    def joint_id(self, visible_id: int, hidden_id: int) -> int:
        key = (int(visible_id), int(hidden_id))
        if key not in self.joint_to_id:
            i = len(self.joint_to_id)
            self.joint_to_id[key] = i
            self.id_to_joint[i] = key
        return int(self.joint_to_id[key])


# ---------------------------------------------------------------------------
# Color-preserving deterministic quotient
# ---------------------------------------------------------------------------
def color_preserving_deterministic_quotient(
    nodes: Set[int],
    edges: Dict[int, Set[int]],
    colors: Dict[int, int],
    counts: Counter,
) -> Tuple[Dict[int, int], Dict, List[Dict]]:
    """Quotient hidden states while preserving visible colors.

    The quotient starts from discrete joint states.  If a block has multiple
    successor blocks of the same visible color, they may be identified.  If a
    block has successors of different visible colors, that ambiguity cannot be
    hidden without erasing visible distinctions; it is counted as a visible-color
    conflict and remains nondeterministic.
    """
    if not nodes:
        return {}, dict(), []
    uf = UnionFind(nodes)
    conflict_sources: Set[int] = set()
    changed = True
    while changed:
        changed = False
        blocks: Dict[int, List[int]] = defaultdict(list)
        for n in nodes:
            blocks[uf.find(n)].append(int(n))
        root_color: Dict[int, int] = {}
        color_conflict_blocks: Set[int] = set()
        for r, members in blocks.items():
            cs = {int(colors.get(m, -1)) for m in members}
            if len(cs) != 1:
                color_conflict_blocks.add(int(r))
                root_color[int(r)] = min(cs) if cs else -1
            else:
                root_color[int(r)] = next(iter(cs))
        for r, members in blocks.items():
            succ_roots = sorted({uf.find(s) for m in members for s in edges.get(m, set())})
            if len(succ_roots) <= 1:
                continue
            by_color: Dict[int, List[int]] = defaultdict(list)
            for sr in succ_roots:
                by_color[int(root_color.get(int(sr), -1))].append(int(sr))
            if len(by_color) > 1:
                conflict_sources.add(int(r))
            for _c, roots in by_color.items():
                if len(roots) > 1:
                    first = roots[0]
                    for other in roots[1:]:
                        if uf.union(first, other):
                            changed = True
    roots = sorted({uf.find(n) for n in nodes})
    root_to_q = {r: i for i, r in enumerate(roots)}
    qmap = {int(n): int(root_to_q[uf.find(n)]) for n in nodes}

    q_counts: Counter = Counter()
    q_colors: Dict[int, int] = {}
    for n, c in counts.items():
        q = int(qmap[int(n)])
        q_counts[q] += int(c)
        q_colors[q] = int(colors.get(int(n), -1))
    q_edges: Dict[int, Set[int]] = defaultdict(set)
    transition_rows: List[Dict] = []
    for a, succs in edges.items():
        qa = int(qmap.get(int(a), -1))
        if qa < 0:
            continue
        for b in succs:
            qb = int(qmap.get(int(b), -1))
            if qb >= 0:
                q_edges[qa].add(qb)

    rec_q = recurrent_nodes(set(q_counts), q_edges) if q_counts else set()
    deterministic_blocks = sum(1 for q in q_counts if len(q_edges.get(int(q), set())) <= 1)
    source_blocks = sum(1 for q in q_counts if q in q_edges)
    visible_moving_edges = 0
    total_edges = 0
    for q, succs in q_edges.items():
        for s in succs:
            total_edges += 1
            if q_colors.get(int(q), -1) != q_colors.get(int(s), -1):
                visible_moving_edges += 1
    for q, c in sorted(q_counts.items()):
        succ = sorted(q_edges.get(int(q), set()))
        transition_rows.append(dict(
            quotient_label=int(q),
            visible_color=int(q_colors.get(int(q), -1)),
            count=int(c),
            frequency=float(c / max(1, sum(q_counts.values()))),
            n_successors=int(len(succ)),
            successors=" ".join(map(str, succ)),
            recurrent=int(int(q) in rec_q),
        ))
    q_entropy = entropy_counts(q_counts.values())
    rec_struct = recurrent_structure_summary(set(q_counts), q_edges, q_counts)
    summary = dict(
        quotient_classes=int(len(q_counts)),
        quotient_entropy_bits=float(q_entropy),
        recurrent_quotient_classes=int(len(rec_q)),
        nontrivial_recurrent_quotient=bool(len(rec_q) > 1),
        deterministic_block_fraction=float(deterministic_blocks / max(1, len(q_counts))),
        source_block_deterministic_fraction=float(sum(1 for q in q_edges if len(q_edges.get(q, set())) <= 1) / max(1, source_blocks)),
        visible_conflict_source_blocks=int(len(conflict_sources)),
        visible_conflict_source_fraction=float(len(conflict_sources) / max(1, source_blocks)),
        quotient_visible_moving_edge_fraction=float(visible_moving_edges / max(1, total_edges)),
        quotient_transition_edges=int(total_edges),
    )
    summary.update(rec_struct)
    return qmap, summary, transition_rows


# ---------------------------------------------------------------------------
# Main hidden-memory audit
# ---------------------------------------------------------------------------
def run_hidden_memory_audit(
    sys: OBG.FiniteRelationalSystem,
    q: int,
    rng: np.random.Generator,
    graph_mode: str = "frame_random_theta",
    max_channel_inputs: int = 256,
    max_channel_backgrounds: int = 256,
    max_state_samples: int = 4096,
    state_warmup: int = 0,
    schedule_samples: int = 16,
    cycle_max_len: int = 6,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    connection_gate: str = "auto",
    gate_min_live_edge_quotients: int = 0,
    gate_min_true_frames: int = 0,
    gate_min_true_transports: int = 0,
    gate_min_cycle_basis: int = 0,
    gate_min_global_chords: int = 0,
    gate_min_global_holonomy: int = 0,
) -> Tuple[object, object, object, object, Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for hidden-memory audit")

    bundle = OSD.build_bundle_objects(
        sys=sys,
        rng=rng,
        max_channel_inputs=max_channel_inputs,
        max_channel_backgrounds=max_channel_backgrounds,
        max_state_samples=max_state_samples,
        state_warmup=state_warmup,
        cycle_max_len=cycle_max_len,
        min_frame_ports=min_frame_ports,
        frame_coordinate_mode=frame_coordinate_mode,
    )
    edges = OSD.live_transport_edges(bundle, require_true=True)
    required_observers = tuple(sorted({int(x) for e in edges for x in e}))

    # Reuse section summary only for connection/cycle-gate metadata.  This also
    # lets users compare hidden-memory output directly with section-dynamics v2.
    _pdf, _tdf, _qdf, section_summary = OSD.run_section_dynamics(
        sys=sys,
        q=q,
        rng=rng,
        max_channel_inputs=max_channel_inputs,
        max_channel_backgrounds=max_channel_backgrounds,
        max_state_samples=max_state_samples,
        state_warmup=state_warmup,
        schedule_samples=max(1, min(int(schedule_samples), 8)),
        cycle_max_len=cycle_max_len,
        min_frame_ports=min_frame_ports,
        frame_coordinate_mode=frame_coordinate_mode,
    )

    gate_mode = str(connection_gate).lower()
    require_gate = gate_mode not in {"off", "none", "false", "0"}
    if gate_mode == "auto":
        targets = default_connection_gate_targets(graph_mode)
        gate_min_live_edge_quotients = int(gate_min_live_edge_quotients or targets["live_edge_quotients"])
        gate_min_true_frames = int(gate_min_true_frames or targets["true_frames"])
        gate_min_true_transports = int(gate_min_true_transports or targets["true_transports"])
        gate_min_cycle_basis = int(gate_min_cycle_basis or targets["cycle_basis"])
        gate_min_global_chords = int(gate_min_global_chords or targets["global_chords"])
        gate_min_global_holonomy = int(gate_min_global_holonomy or targets["global_holonomy"])
    gate_passed, gate_progress, gate_details = connection_gate_status(
        section_summary,
        gate_min_live_edge_quotients=gate_min_live_edge_quotients,
        gate_min_true_frames=gate_min_true_frames,
        gate_min_true_transports=gate_min_true_transports,
        gate_min_cycle_basis=gate_min_cycle_basis,
        gate_min_global_chords=gate_min_global_chords,
        gate_min_global_holonomy=gate_min_global_holonomy,
    )

    states = all_or_sample_states(q, sys.k, max_state_samples, rng)
    orders = OSD.schedule_orders(sys.k, schedule_samples, rng)
    enc = Encoders.empty()

    current_reps: Dict[Tuple[int, ...], Tuple[int, int, int]] = {}
    next_reps: Dict[Tuple[int, ...], Tuple[int, int, int]] = {}

    def reps_for_state(state: Sequence[int], cache: Dict[Tuple[int, ...], Tuple[int, int, int]]) -> Tuple[int, int, int]:
        key = tuple(int(x) for x in state)
        if key not in cache:
            v, h = state_visible_hidden(sys, key, bundle, required_observers)
            vid = enc.visible_id(v)
            hid = enc.hidden_id(h)
            jid = enc.joint_id(vid, hid)
            cache[key] = (vid, hid, jid)
        return cache[key]

    visible_counts: Counter = Counter()
    hidden_counts: Counter = Counter()
    joint_counts: Counter = Counter()
    visible_transitions: Counter = Counter()
    joint_to_visible_transitions: Counter = Counter()
    joint_transitions: Counter = Counter()
    observer_rows_accum: Dict[int, Dict[str, Counter]] = {}

    # Observer-local transition counters.
    for oid in required_observers:
        observer_rows_accum[int(oid)] = dict(
            visible_counts=Counter(), joint_counts=Counter(),
            visible_transitions=Counter(), hidden_visible_transitions=Counter(),
        )

    for s in states:
        state = tuple(int(x) for x in s)
        vid, hid, jid = reps_for_state(state, current_reps)
        visible_counts[vid] += 1
        hidden_counts[hid] += 1
        joint_counts[jid] += 1

        # Per-observer current keys.
        per_obs_cur: Dict[int, Tuple[Tuple[str, int], int]] = {}
        for oid in required_observers:
            sym = observer_visible_symbol(sys, state, bundle, int(oid))
            hloc = internal_code(sys, state, bundle.comps[int(oid)])
            per_obs_cur[int(oid)] = (sym, int(hloc))
            observer_rows_accum[int(oid)]["visible_counts"][sym] += 1
            observer_rows_accum[int(oid)]["joint_counts"][(sym, int(hloc))] += 1

        for order in orders:
            nxt = OSD.sequential_step(sys, state, order)
            nvid, nhid, njid = reps_for_state(nxt, next_reps)
            visible_transitions[(vid, nvid)] += 1
            joint_to_visible_transitions[(jid, nvid)] += 1
            joint_transitions[(jid, njid)] += 1
            for oid in required_observers:
                sym0, hloc0 = per_obs_cur[int(oid)]
                sym1 = observer_visible_symbol(sys, nxt, bundle, int(oid))
                observer_rows_accum[int(oid)]["visible_transitions"][(sym0, sym1)] += 1
                observer_rows_accum[int(oid)]["hidden_visible_transitions"][((sym0, int(hloc0)), sym1)] += 1

    visible_acc, visible_moving, _vsrc, vtrans_total = transition_accuracy(visible_transitions)
    hidden_acc, _hidden_moving_target_visible, _jsrc, jvis_total = transition_accuracy(joint_to_visible_transitions)
    joint_acc, joint_moving, _jsrc2, jtrans_total = transition_accuracy(joint_transitions)

    nodes = set(int(k) for k in joint_counts)
    for (a, b), _c in joint_transitions.items():
        nodes.add(int(a)); nodes.add(int(b))
    color = {int(j): int(enc.id_to_joint[int(j)][0]) for j in nodes if int(j) in enc.id_to_joint}
    qmap, quotient_summary, quotient_rows = color_preserving_deterministic_quotient(
        nodes=nodes,
        edges=transition_edges(joint_transitions),
        colors=color,
        counts=joint_counts,
    )

    # Visible quotienting correction.  First quotient the visible dynamics alone:
    # if schedules send one visible state to several visible successors, merge
    # those successor visible states until the visible quotient transition is
    # deterministic.  Then do the same on the full joint (visible, hidden) state
    # *without* preserving visible colors, and inspect what visible distinctions
    # survive.
    visible_nodes = set(int(v) for v in visible_counts)
    visible_qmap, visible_q_summary, visible_q_rows = deterministic_schedule_quotient(
        nodes=visible_nodes,
        edges=transition_edges(visible_transitions),
        counts=visible_counts,
        colors=None,
    )
    joint_schedule_qmap, joint_schedule_q_summary, joint_schedule_q_rows = deterministic_schedule_quotient(
        nodes=nodes,
        edges=transition_edges(joint_transitions),
        counts=joint_counts,
        colors=color,
    )
    induced_visible_qmap, induced_visible_q_summary, induced_visible_q_rows = visible_partition_induced_by_joint_quotient(
        joint_qmap=joint_schedule_qmap,
        id_to_joint=enc.id_to_joint,
        visible_counts=visible_counts,
        visible_transitions=visible_transitions,
    )

    q_counts: Counter = Counter()
    for j, c in joint_counts.items():
        q_counts[int(qmap.get(int(j), -1))] += int(c)
    q_classes_by_visible: Dict[int, Set[int]] = defaultdict(set)
    hidden_classes_by_visible: Dict[int, Set[int]] = defaultdict(set)
    joint_schedule_q_by_visible: Dict[int, Set[int]] = defaultdict(set)
    for jid, (vid, hid) in enc.id_to_joint.items():
        if jid in joint_counts:
            hidden_classes_by_visible[int(vid)].add(int(hid))
            if jid in qmap:
                q_classes_by_visible[int(vid)].add(int(qmap[jid]))
            if jid in joint_schedule_qmap:
                joint_schedule_q_by_visible[int(vid)].add(int(joint_schedule_qmap[jid]))

    distinct_visible = len(visible_counts)
    distinct_hidden = len(hidden_counts)
    distinct_joint = len(joint_counts)
    quotient_classes = int(quotient_summary.get("quotient_classes", 0))
    extra_memory_classes = max(0, quotient_classes - distinct_visible)
    possible_extra = max(1, distinct_joint - distinct_visible)
    memory_compression_ratio = float(math.log2(max(1, quotient_classes)) / max(1e-12, math.log2(max(2, distinct_joint))))
    memory_extra_fraction = float(extra_memory_classes / possible_extra)
    ambiguity_reduction = float((hidden_acc - visible_acc) / max(1e-12, 1.0 - visible_acc)) if hidden_acc >= visible_acc else 0.0

    visible_schedule_classes = int(visible_q_summary.get("quotient_classes", 0))
    joint_schedule_classes = int(joint_schedule_q_summary.get("quotient_classes", 0))
    induced_visible_classes = int(induced_visible_q_summary.get("quotient_classes", 0))
    visible_schedule_retention = float(math.log2(max(1, visible_schedule_classes)) / max(1e-12, math.log2(max(2, distinct_visible)))) if distinct_visible > 1 else 0.0
    joint_schedule_retention = float(math.log2(max(1, joint_schedule_classes)) / max(1e-12, math.log2(max(2, distinct_joint)))) if distinct_joint > 1 else 0.0
    induced_visible_retention = float(math.log2(max(1, induced_visible_classes)) / max(1e-12, math.log2(max(2, distinct_visible)))) if distinct_visible > 1 else 0.0
    joint_schedule_compression = float(1.0 - joint_schedule_retention)

    # Pattern rows for sampled current joint states.
    pattern_rows: List[Dict] = []
    total_state_count = int(sum(joint_counts.values()))
    for jid, c in sorted(joint_counts.items()):
        vid, hid = enc.id_to_joint[int(jid)]
        vtuple = enc.id_to_visible[int(vid)]
        pattern_rows.append(dict(
            joint_id=int(jid),
            visible_id=int(vid),
            hidden_id=int(hid),
            quotient_label=int(qmap.get(int(jid), -1)),
            count=int(c),
            frequency=float(c / max(1, total_state_count)),
            visible_signature=json.dumps(vtuple),
            hidden_tuple=" ".join(map(str, enc.id_to_hidden[int(hid)])),
        ))

    transition_rows: List[Dict] = []
    for (a, b), c in sorted(joint_transitions.items()):
        qa = int(qmap.get(int(a), -1)); qb = int(qmap.get(int(b), -1))
        va = int(enc.id_to_joint[int(a)][0]) if int(a) in enc.id_to_joint else -1
        vb = int(enc.id_to_joint[int(b)][0]) if int(b) in enc.id_to_joint else -1
        transition_rows.append(dict(
            source_joint=int(a), target_joint=int(b), count=int(c),
            source_visible=va, target_visible=vb,
            source_quotient=qa, target_quotient=qb,
            visible_moved=int(va != vb),
            quotient_moved=int(qa != qb),
        ))

    observer_rows: List[Dict] = []
    for oid, d in sorted(observer_rows_accum.items()):
        vacc, vmov, _s, _t = transition_accuracy(d["visible_transitions"])
        hacc, _hmov, _hs, _ht = transition_accuracy(d["hidden_visible_transitions"])
        vcnt = d["visible_counts"]
        jcnt = d["joint_counts"]
        obs_gain = float((hacc - vacc) / max(1e-12, 1.0 - vacc)) if hacc >= vacc else 0.0
        observer_rows.append(dict(
            observer_id=int(oid),
            n_internal_vertices=int(len(bundle.comps[int(oid)])),
            n_visible_symbols=int(len(vcnt)),
            n_hidden_conditioned_symbols=int(len(jcnt)),
            visible_entropy_bits=float(entropy_counts(vcnt.values())),
            hidden_conditioned_entropy_bits=float(entropy_counts(jcnt.values())),
            visible_transition_determinism_accuracy=float(vacc),
            hidden_conditioned_visible_determinism_accuracy=float(hacc),
            ambiguity_reduction=float(obs_gain),
            visible_moving_transition_fraction=float(vmov),
            compression_raw_states=int(q ** len(bundle.comps[int(oid)])),
        ))

    hidden_memory_signal = bool(
        gate_passed and hidden_acc > visible_acc + 1e-9 and ambiguity_reduction > 0.10 and quotient_classes > distinct_visible
    )
    schedule_absorption_signal = bool(
        gate_passed and hidden_acc >= 0.999 and quotient_summary.get("source_block_deterministic_fraction", 0.0) >= 0.999
        and quotient_classes > distinct_visible and quotient_classes < distinct_joint
    )

    visible_schedule_quotient_signal = bool(
        gate_passed
        and visible_schedule_classes > 1
        and visible_schedule_classes < max(1, distinct_visible)
        and visible_q_summary.get("source_block_deterministic_fraction", 0.0) >= 0.999
    )
    joint_schedule_quotient_signal = bool(
        gate_passed
        and joint_schedule_classes > 1
        and joint_schedule_classes < max(1, distinct_joint)
        and induced_visible_classes > 1
        and joint_schedule_q_summary.get("source_block_deterministic_fraction", 0.0) >= 0.999
    )
    visible_quotient_schedule_absorption_signal = bool(visible_schedule_quotient_signal or joint_schedule_quotient_signal)

    memory_periodic = bool(quotient_summary.get("nontrivial_periodic_recurrence", False))
    visible_periodic = bool(visible_q_summary.get("nontrivial_periodic_recurrence", False))
    joint_periodic = bool(joint_schedule_q_summary.get("nontrivial_periodic_recurrence", False))
    induced_visible_periodic = bool(induced_visible_q_summary.get("nontrivial_periodic_recurrence", False))

    dynamics_candidate = bool(
        (schedule_absorption_signal
         and (quotient_summary.get("nontrivial_recurrent_quotient", False) or memory_periodic)
         and quotient_summary.get("quotient_visible_moving_edge_fraction", 0.0) > 0)
        or (visible_schedule_quotient_signal
            and (visible_q_summary.get("nontrivial_recurrent_quotient", False) or visible_periodic)
            and visible_q_summary.get("quotient_moving_edge_fraction", 0.0) > 0)
        or (joint_schedule_quotient_signal
            and (joint_schedule_q_summary.get("nontrivial_recurrent_quotient", False) or joint_periodic or induced_visible_periodic)
            and induced_visible_q_summary.get("moving_transition_fraction", 0.0) > 0)
    )

    if require_gate and not gate_passed:
        verdict = "HIDDEN MEMORY AUDIT BLOCKED: observer-frame connection gate failed"
    elif dynamics_candidate:
        verdict = "VISIBLE-QUOTIENT PERIODIC/RECURRENT DYNAMICS CANDIDATE: schedule-consistent quotient has nontrivial recurrence"
    elif visible_quotient_schedule_absorption_signal:
        verdict = "VISIBLE-QUOTIENT SCHEDULE-ABSORPTION SIGNAL: visible distinctions quotient to schedule-consistent dynamics"
    elif schedule_absorption_signal:
        verdict = "HIDDEN MEMORY SCHEDULE-ABSORPTION SIGNAL: compressed observer memory makes visible dynamics schedule-deterministic"
    elif hidden_memory_signal:
        verdict = "HIDDEN MEMORY AMBIGUITY-REDUCTION SIGNAL: observer-internal state improves visible schedule consistency"
    elif hidden_acc > visible_acc + 1e-9:
        verdict = "WEAK HIDDEN MEMORY SIGNAL: internal state improves prediction but not enough for autonomous dynamics"
    else:
        verdict = "NO HIDDEN MEMORY / VISIBLE-QUOTIENT SCHEDULE-ABSORPTION SIGNAL in this regime"

    # Combine quotient row families in one CSV, with a type tag.
    combined_quotient_rows: List[Dict] = []
    for r in quotient_rows:
        rr = dict(r); rr.setdefault("quotient_type", "color_preserving_memory"); combined_quotient_rows.append(rr)
    for r in visible_q_rows:
        rr = dict(r); rr.setdefault("quotient_type", "visible_schedule"); combined_quotient_rows.append(rr)
    for r in joint_schedule_q_rows:
        rr = dict(r); rr.setdefault("quotient_type", "joint_schedule_visible_quotient"); combined_quotient_rows.append(rr)
    for r in induced_visible_q_rows:
        rr = dict(r); rr.setdefault("quotient_type", "induced_visible_from_joint_schedule"); combined_quotient_rows.append(rr)

    summary = dict(
        verdict=verdict,
        audit_version="observer_hidden_memory_v3_visible_quotient_period_audit",
        q=int(q),
        n_vertices=int(sys.k),
        n_observers=int(len(bundle.comps)),
        n_required_observers=int(len(required_observers)),
        required_observers=" ".join(map(str, required_observers)),
        n_live_transport_edges=int(len(edges)),
        live_transport_edges=[f"{a}->{b}" for a, b in edges],
        frame_coordinate_mode=str(frame_coordinate_mode),
        graph_mode=str(graph_mode),
        n_sampled_states=int(len(states)),
        n_schedules=int(len(orders)),
        connection_gate_required=bool(require_gate),
        connection_gate_passed=bool(gate_passed),
        connection_gate_progress=float(gate_progress),
        connection_gate_details=gate_details,
        bundle_summary=bundle.summary,
        section_summary_key=dict(
            cycle_basis_size=int(section_summary.get("cycle_basis_size", 0)),
            connection_holonomy_bits=str(section_summary.get("connection_holonomy_bits", "")),
            global_chord_count=int(section_summary.get("global_chord_count", 0)),
            global_valid_holonomy=int(section_summary.get("global_valid_holonomy", 0)),
            global_nontrivial_holonomy=int(section_summary.get("global_nontrivial_holonomy", 0)),
            global_generated_group=str(section_summary.get("global_generated_group", "none")),
            coherence_defect_transition_determinism_accuracy=float(section_summary.get("coherence_defect_transition_determinism_accuracy", 0.0)),
            coherence_defect_recurrent_quotient_classes=int(section_summary.get("coherence_defect_recurrent_quotient_classes", 0)),
        ),
        distinct_visible_states=int(distinct_visible),
        distinct_hidden_states=int(distinct_hidden),
        distinct_joint_states=int(distinct_joint),
        visible_entropy_bits=float(entropy_counts(visible_counts.values())),
        hidden_entropy_bits=float(entropy_counts(hidden_counts.values())),
        joint_entropy_bits=float(entropy_counts(joint_counts.values())),
        visible_only_transition_determinism_accuracy=float(visible_acc),
        hidden_conditioned_visible_determinism_accuracy=float(hidden_acc),
        full_joint_transition_determinism_accuracy=float(joint_acc),
        visible_moving_transition_fraction=float(visible_moving),
        full_joint_moving_transition_fraction=float(joint_moving),
        hidden_memory_ambiguity_reduction=float(ambiguity_reduction),
        hidden_memory_gain=float(hidden_acc - visible_acc),
        hidden_memory_signal=bool(hidden_memory_signal),
        schedule_absorption_signal=bool(schedule_absorption_signal),
        dynamics_candidate=bool(dynamics_candidate),
        memory_quotient_classes=int(quotient_classes),
        memory_quotient_entropy_bits=float(quotient_summary.get("quotient_entropy_bits", 0.0)),
        memory_quotient_recurrent_classes=int(quotient_summary.get("recurrent_quotient_classes", 0)),
        memory_quotient_nontrivial_recurrent=bool(quotient_summary.get("nontrivial_recurrent_quotient", False)),
        memory_quotient_recurrent_component_count=int(quotient_summary.get("recurrent_component_count", 0)),
        memory_quotient_recurrent_state_count=int(quotient_summary.get("recurrent_state_count", 0)),
        memory_quotient_recurrent_component_sizes=quotient_summary.get("recurrent_component_sizes", []),
        memory_quotient_cycle_lengths=quotient_summary.get("recurrent_cycle_lengths", []),
        memory_quotient_max_period=int(quotient_summary.get("max_recurrent_period", 0)),
        memory_quotient_fixed_point_count=int(quotient_summary.get("fixed_point_count", 0)),
        memory_quotient_nontrivial_periodic_recurrence=bool(quotient_summary.get("nontrivial_periodic_recurrence", False)),
        memory_quotient_deterministic_block_fraction=float(quotient_summary.get("deterministic_block_fraction", 0.0)),
        memory_quotient_source_block_deterministic_fraction=float(quotient_summary.get("source_block_deterministic_fraction", 0.0)),
        memory_quotient_visible_conflict_source_fraction=float(quotient_summary.get("visible_conflict_source_fraction", 0.0)),
        memory_quotient_visible_moving_edge_fraction=float(quotient_summary.get("quotient_visible_moving_edge_fraction", 0.0)),
        memory_quotient_transition_edges=int(quotient_summary.get("quotient_transition_edges", 0)),
        memory_extra_classes=int(extra_memory_classes),
        memory_extra_fraction=float(memory_extra_fraction),
        memory_compression_ratio=float(memory_compression_ratio),
        visible_schedule_quotient_signal=bool(visible_schedule_quotient_signal),
        joint_schedule_quotient_signal=bool(joint_schedule_quotient_signal),
        visible_quotient_schedule_absorption_signal=bool(visible_quotient_schedule_absorption_signal),
        visible_schedule_quotient_classes=int(visible_schedule_classes),
        visible_schedule_quotient_entropy_bits=float(visible_q_summary.get("quotient_entropy_bits", 0.0)),
        visible_schedule_quotient_retention_fraction=float(visible_schedule_retention),
        visible_schedule_quotient_recurrent_classes=int(visible_q_summary.get("recurrent_quotient_classes", 0)),
        visible_schedule_quotient_nontrivial_recurrent=bool(visible_q_summary.get("nontrivial_recurrent_quotient", False)),
        visible_schedule_quotient_recurrent_component_count=int(visible_q_summary.get("recurrent_component_count", 0)),
        visible_schedule_quotient_recurrent_state_count=int(visible_q_summary.get("recurrent_state_count", 0)),
        visible_schedule_quotient_recurrent_component_sizes=visible_q_summary.get("recurrent_component_sizes", []),
        visible_schedule_quotient_cycle_lengths=visible_q_summary.get("recurrent_cycle_lengths", []),
        visible_schedule_quotient_max_period=int(visible_q_summary.get("max_recurrent_period", 0)),
        visible_schedule_quotient_fixed_point_count=int(visible_q_summary.get("fixed_point_count", 0)),
        visible_schedule_quotient_nontrivial_periodic_recurrence=bool(visible_q_summary.get("nontrivial_periodic_recurrence", False)),
        visible_schedule_quotient_source_block_deterministic_fraction=float(visible_q_summary.get("source_block_deterministic_fraction", 0.0)),
        visible_schedule_quotient_moving_edge_fraction=float(visible_q_summary.get("quotient_moving_edge_fraction", 0.0)),
        visible_schedule_quotient_transition_edges=int(visible_q_summary.get("quotient_transition_edges", 0)),
        joint_schedule_quotient_classes=int(joint_schedule_classes),
        joint_schedule_quotient_entropy_bits=float(joint_schedule_q_summary.get("quotient_entropy_bits", 0.0)),
        joint_schedule_quotient_retention_fraction=float(joint_schedule_retention),
        joint_schedule_quotient_compression_fraction=float(joint_schedule_compression),
        joint_schedule_quotient_recurrent_classes=int(joint_schedule_q_summary.get("recurrent_quotient_classes", 0)),
        joint_schedule_quotient_nontrivial_recurrent=bool(joint_schedule_q_summary.get("nontrivial_recurrent_quotient", False)),
        joint_schedule_quotient_recurrent_component_count=int(joint_schedule_q_summary.get("recurrent_component_count", 0)),
        joint_schedule_quotient_recurrent_state_count=int(joint_schedule_q_summary.get("recurrent_state_count", 0)),
        joint_schedule_quotient_recurrent_component_sizes=joint_schedule_q_summary.get("recurrent_component_sizes", []),
        joint_schedule_quotient_cycle_lengths=joint_schedule_q_summary.get("recurrent_cycle_lengths", []),
        joint_schedule_quotient_max_period=int(joint_schedule_q_summary.get("max_recurrent_period", 0)),
        joint_schedule_quotient_fixed_point_count=int(joint_schedule_q_summary.get("fixed_point_count", 0)),
        joint_schedule_quotient_nontrivial_periodic_recurrence=bool(joint_schedule_q_summary.get("nontrivial_periodic_recurrence", False)),
        joint_schedule_quotient_source_block_deterministic_fraction=float(joint_schedule_q_summary.get("source_block_deterministic_fraction", 0.0)),
        joint_schedule_quotient_visible_mixed_block_fraction=float(joint_schedule_q_summary.get("visible_mixed_block_fraction", 0.0)),
        joint_schedule_quotient_mean_visible_colors_per_block=float(joint_schedule_q_summary.get("mean_visible_colors_per_block", 0.0)),
        joint_schedule_quotient_visible_moving_edge_fraction=float(joint_schedule_q_summary.get("quotient_visible_moving_edge_fraction", 0.0)),
        induced_visible_quotient_classes=int(induced_visible_classes),
        induced_visible_quotient_entropy_bits=float(induced_visible_q_summary.get("quotient_entropy_bits", 0.0)),
        induced_visible_quotient_retention_fraction=float(induced_visible_retention),
        induced_visible_quotient_transition_determinism_accuracy=float(induced_visible_q_summary.get("transition_determinism_accuracy", 0.0)),
        induced_visible_quotient_moving_transition_fraction=float(induced_visible_q_summary.get("moving_transition_fraction", 0.0)),
        induced_visible_quotient_recurrent_component_count=int(induced_visible_q_summary.get("recurrent_component_count", 0)),
        induced_visible_quotient_recurrent_state_count=int(induced_visible_q_summary.get("recurrent_state_count", 0)),
        induced_visible_quotient_recurrent_component_sizes=induced_visible_q_summary.get("recurrent_component_sizes", []),
        induced_visible_quotient_cycle_lengths=induced_visible_q_summary.get("recurrent_cycle_lengths", []),
        induced_visible_quotient_max_period=int(induced_visible_q_summary.get("max_recurrent_period", 0)),
        induced_visible_quotient_fixed_point_count=int(induced_visible_q_summary.get("fixed_point_count", 0)),
        induced_visible_quotient_nontrivial_periodic_recurrence=bool(induced_visible_q_summary.get("nontrivial_periodic_recurrence", False)),
        mean_hidden_classes_per_visible=float(np.mean([len(v) for v in hidden_classes_by_visible.values()])) if hidden_classes_by_visible else 0.0,
        mean_memory_classes_per_visible=float(np.mean([len(v) for v in q_classes_by_visible.values()])) if q_classes_by_visible else 0.0,
        max_hidden_classes_per_visible=int(max([len(v) for v in hidden_classes_by_visible.values()] or [0])),
        max_memory_classes_per_visible=int(max([len(v) for v in q_classes_by_visible.values()] or [0])),
    )
    return pd.DataFrame(pattern_rows), pd.DataFrame(transition_rows), pd.DataFrame(combined_quotient_rows), pd.DataFrame(observer_rows), summary


# ---------------------------------------------------------------------------
# CLI / I/O
# ---------------------------------------------------------------------------
def load_system_from_winners(path: str, index: int = 0) -> OBG.FiniteRelationalSystem:
    return OSD.load_system_from_winners(path, index)


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


def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def write_outputs(pattern_df, transition_df, quotient_df, observer_df, summary: Dict, out: str) -> None:
    pattern_df.to_csv(out, index=False)
    transition_df.to_csv(derived_path(out, "_transitions"), index=False)
    quotient_df.to_csv(derived_path(out, "_quotient"), index=False)
    observer_df.to_csv(derived_path(out, "_observers"), index=False)
    with open(derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def make_plot(pattern_df, transition_df, quotient_df, observer_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle(summary.get("verdict", "observer hidden memory"), fontsize=10)
    labels = ["visible only", "hidden conditioned", "memory quotient"]
    vals = [
        float(summary.get("visible_only_transition_determinism_accuracy", 0.0)),
        float(summary.get("hidden_conditioned_visible_determinism_accuracy", 0.0)),
        float(summary.get("memory_quotient_source_block_deterministic_fraction", 0.0)),
    ]
    axes[0].bar(labels, vals)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("schedule determinism")
    axes[0].tick_params(axis='x', rotation=20)
    if len(observer_df):
        axes[1].bar(observer_df["observer_id"].astype(str), observer_df["ambiguity_reduction"].astype(float))
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("per-observer gain")
    if len(pattern_df):
        axes[2].hist(pattern_df["quotient_label"].astype(int), bins=max(1, int(pattern_df["quotient_label"].max()) + 1))
    axes[2].set_title("memory quotient labels")
    axes[2].set_xlabel("label")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Observer hidden-memory schedule-absorption audit")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--winners", default="", help="optional winners pickle; if supplied, graph-mode is ignored")
    p.add_argument("--winner-index", type=int, default=0)
    p.add_argument("--graph-mode", choices=[
        "frame_twist_c2", "frame_flat_c2", "frame_random_diamond",
        "frame_theta_flat_c2", "frame_theta_twist_c2", "frame_random_theta",
        "frame_double_flat_c2", "frame_double_twist_c2", "frame_random_double_diamond",
        "componented", "er"], default="frame_random_theta")
    p.add_argument("--vertices", type=int, default=11)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.16)
    p.add_argument("--inter-prob", type=float, default=0.35)
    p.add_argument("--extra-intra-prob", type=float, default=0.25)
    p.add_argument("--max-pred", type=int, default=4)
    p.add_argument("--max-channel-inputs", type=int, default=128)
    p.add_argument("--max-channel-backgrounds", type=int, default=128)
    p.add_argument("--max-state-samples", type=int, default=2048)
    p.add_argument("--state-warmup", type=int, default=0)
    p.add_argument("--schedule-samples", type=int, default=8)
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--min-frame-ports", type=int, default=2)
    p.add_argument("--frame-coordinate-mode", choices=["local_charts", "common_factor"], default="local_charts")
    p.add_argument("--connection-gate", choices=["auto", "off"], default="auto")
    p.add_argument("--gate-min-live-edge-quotients", type=int, default=0)
    p.add_argument("--gate-min-true-frames", type=int, default=0)
    p.add_argument("--gate-min-true-transports", type=int, default=0)
    p.add_argument("--gate-min-cycle-basis", type=int, default=0)
    p.add_argument("--gate-min-global-chords", type=int, default=0)
    p.add_argument("--gate-min-global-holonomy", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/observer_hidden_memory.csv")
    p.add_argument("--plot", default="example_results/fig_observer_hidden_memory.png")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rng = np.random.default_rng(int(args.seed))
    sys = make_system_from_args(args, rng)
    pattern_df, transition_df, quotient_df, observer_df, summary = run_hidden_memory_audit(
        sys=sys,
        q=int(args.q),
        rng=rng,
        graph_mode=str(args.graph_mode),
        max_channel_inputs=int(args.max_channel_inputs),
        max_channel_backgrounds=int(args.max_channel_backgrounds),
        max_state_samples=int(args.max_state_samples),
        state_warmup=int(args.state_warmup),
        schedule_samples=int(args.schedule_samples),
        cycle_max_len=int(args.cycle_max_len),
        min_frame_ports=int(args.min_frame_ports),
        frame_coordinate_mode=str(args.frame_coordinate_mode),
        connection_gate=str(args.connection_gate),
        gate_min_live_edge_quotients=int(args.gate_min_live_edge_quotients),
        gate_min_true_frames=int(args.gate_min_true_frames),
        gate_min_true_transports=int(args.gate_min_true_transports),
        gate_min_cycle_basis=int(args.gate_min_cycle_basis),
        gate_min_global_chords=int(args.gate_min_global_chords),
        gate_min_global_holonomy=int(args.gate_min_global_holonomy),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_outputs(pattern_df, transition_df, quotient_df, observer_df, summary, args.out)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plot(pattern_df, transition_df, quotient_df, observer_df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {derived_path(args.out, '_transitions')}")
    print(f"wrote {derived_path(args.out, '_quotient')}")
    print(f"wrote {derived_path(args.out, '_observers')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
