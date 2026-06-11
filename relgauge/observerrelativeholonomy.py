"""
observerrelativeholonomy.py

Observer-relative holonomy audit for finite relational systems.

This module is the next layer after observerboundarygeometry.py.  The previous
module discovers an observer-relative cut graph: SCC observers are nodes,
live inter-SCC boundary channels are edges, and simple cycles in that graph are
candidate plaquettes discovered from inside the closed system rather than drawn
externally.

This module asks whether those discovered cycles carry transported quotient
labels and path holonomy.

Important scope
---------------
This is an audit, not a selection experiment.  It does not reward C2, C3,
nonabelian groups, or holonomy.  It generates closed finite systems, discovers
observer-relative cycles, then attempts to extract exact finite common factors
on the inter-observer channels and compose them around diamond-like directed
cycles.  Holonomy/group information is post-hoc.

A simple observer cycle is usable for holonomy when the directed inter-SCC
orientation on the cycle has one source and one sink and hence two directed
paths from source to sink.  This is the observer-relative analogue of the
shared-square/two-path holonomy test.

Example
-------
python -m relgauge.observerrelativeholonomy 4 ^
  --graphs 100 --vertices 18 --graph-mode er --edge-prob 0.16 ^
  --capacity-threshold 0.05 --cycle-capacity-threshold 0.1 ^
  --out example_results/observer_relative_holonomy_q4.csv ^
  --plot example_results/fig_observer_relative_holonomy_q4.png
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

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import observerboundarygeometry as OBG
except Exception:  # pragma: no cover
    import observerboundarygeometry as OBG  # type: ignore

Edge = Tuple[int, int]
CompEdge = Tuple[int, int]


# ---------------------------------------------------------------------------
# Generic finite helpers
# ---------------------------------------------------------------------------
def entropy_from_counts(counts: Sequence[int]) -> float:
    arr = np.asarray(counts, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    p = arr[arr > 0] / total
    return float(-(p * np.log2(p)).sum())


def mutual_information_from_joint(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=float)
    if joint.sum() <= 0:
        return 0.0
    return entropy_from_counts(joint.sum(axis=1)) + entropy_from_counts(joint.sum(axis=0)) - entropy_from_counts(joint.ravel())


def encode_tuple(vals: Sequence[int], q: int) -> int:
    code = 0
    for v in vals:
        code = code * q + int(v)
    return int(code)


def all_or_sample_assignments(q: int, nodes: Sequence[int], max_configs: int, rng: np.random.Generator) -> List[Tuple[int, ...]]:
    nodes = list(nodes)
    if len(nodes) == 0:
        return [tuple()]
    total = q ** len(nodes)
    if total <= max_configs:
        return list(itertools.product(range(q), repeat=len(nodes)))
    return [tuple(int(x) for x in rng.integers(0, q, size=len(nodes))) for _ in range(max_configs)]


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
# Quotient extraction on local observer-boundary channels
# ---------------------------------------------------------------------------
@dataclass
class EdgeQuotient:
    comp_edge: CompEdge
    micro_edges: Tuple[Edge, ...]
    source_nodes: Tuple[int, ...]
    target_nodes: Tuple[int, ...]
    source_label: Dict[int, int]
    target_label: Dict[int, int]
    n_classes: int
    residual_bits: float
    residual_qcoords: float
    label_entropy_bits: float
    transport_mi_bits: float
    transport_mi_norm: float
    deterministic_accuracy: float
    live: bool


def _relation_pairs_for_channel(
    sys: OBG.FiniteRelationalSystem,
    input_nodes: Sequence[int],
    target_nodes: Sequence[int],
    max_input_configs: int,
    max_background_configs: int,
    rng: np.random.Generator,
) -> Tuple[Set[Tuple[int, int]], Counter, Counter, np.ndarray, int, int]:
    """Observed relation between input codes and updated target codes."""
    q = sys.q
    input_nodes = tuple(sorted(set(int(x) for x in input_nodes)))
    target_nodes = tuple(sorted(set(int(x) for x in target_nodes)))
    if not input_nodes or not target_nodes:
        return set(), Counter(), Counter(), np.zeros((0, 0), dtype=np.int64), 0, 0

    needed: Set[int] = set()
    for v in target_nodes:
        needed.update(sys.preds[v])
    bg_nodes = tuple(sorted(needed - set(input_nodes)))

    input_assignments = all_or_sample_assignments(q, input_nodes, max_input_configs, rng)
    bg_assignments = all_or_sample_assignments(q, bg_nodes, max_background_configs, rng)
    x_codes = sorted(set(encode_tuple(a, q) for a in input_assignments))
    x_to_row = {c: i for i, c in enumerate(x_codes)}
    n_y = q ** len(target_nodes)
    joint = np.zeros((len(x_codes), n_y), dtype=np.int64)
    rel: Set[Tuple[int, int]] = set()
    x_counts: Counter = Counter()
    y_counts: Counter = Counter()

    for ia in input_assignments:
        assignment = {node: val for node, val in zip(input_nodes, ia)}
        x_code = encode_tuple(ia, q)
        row = x_to_row[x_code]
        for ba in bg_assignments:
            a2 = dict(assignment)
            a2.update({node: val for node, val in zip(bg_nodes, ba)})
            y_vals = [sys.eval_vertex_from_assignment(v, a2) for v in target_nodes]
            y_code = encode_tuple(y_vals, q)
            rel.add((x_code, y_code))
            x_counts[x_code] += 1
            y_counts[y_code] += 1
            joint[row, y_code] += 1
    return rel, x_counts, y_counts, joint, q ** len(input_nodes), n_y


def edge_common_factor(
    sys: OBG.FiniteRelationalSystem,
    comp_edge: CompEdge,
    micro_edges: Sequence[Edge],
    max_input_configs: int,
    max_background_configs: int,
    rng: np.random.Generator,
    min_classes: int = 2,
    min_label_entropy: float = 0.25,
    min_transport_mi_norm: float = 0.95,
) -> EdgeQuotient:
    """Maximal exact common factor of an inter-observer channel relation.

    Build the bipartite graph X--Y of observed input/output pairs.  Connected
    components are the maximal exact labels common to X and Y.  A nontrivial
    transported quotient exists if there are at least two used components with
    nonzero label entropy.
    """
    q = sys.q
    micro_edges = tuple((int(u), int(v)) for u, v in micro_edges)
    source_nodes = tuple(sorted({u for u, _ in micro_edges}))
    target_nodes = tuple(sorted({v for _, v in micro_edges}))
    rel, x_counts, y_counts, _joint, n_x_total, n_y_total = _relation_pairs_for_channel(
        sys, source_nodes, target_nodes, max_input_configs, max_background_configs, rng
    )
    if not rel:
        return EdgeQuotient(comp_edge, micro_edges, source_nodes, target_nodes, {}, {}, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    x_codes = sorted({x for x, _ in rel})
    y_codes = sorted({y for _, y in rel})
    x_index = {x: i for i, x in enumerate(x_codes)}
    y_index = {y: i + len(x_codes) for i, y in enumerate(y_codes)}
    uf = UnionFind(len(x_codes) + len(y_codes))
    for x, y in rel:
        uf.union(x_index[x], y_index[y])

    roots = sorted(set(uf.find(i) for i in range(len(x_codes) + len(y_codes))))
    root_to_lab = {r: i for i, r in enumerate(roots)}
    source_label = {x: root_to_lab[uf.find(x_index[x])] for x in x_codes}
    target_label = {y: root_to_lab[uf.find(y_index[y])] for y in y_codes}
    n_classes = len(roots)

    # Label joint over observed channel samples.  Because labels are exact
    # components, mass should lie on diagonal after relabeling.
    lab_joint = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in rel:
        # weight by product of marginal counts as a stable proxy for repeated samples
        w = max(1, min(int(x_counts[x]), int(y_counts[y])))
        lab_joint[source_label[x], target_label[y]] += w
    h_label = entropy_from_counts(lab_joint.sum(axis=1))
    mi = mutual_information_from_joint(lab_joint)
    mi_norm = float(mi / h_label) if h_label > 1e-12 else 0.0
    residual_bits = math.log2(max(1, n_classes))
    residual_qcoords = residual_bits / math.log2(q) if q > 1 else 0.0

    # Best deterministic label map accuracy X_label -> Y_label.
    row_sums = lab_joint.sum(axis=1)
    acc = float(lab_joint.max(axis=1).sum() / max(1, row_sums.sum())) if lab_joint.size else 0.0
    live = bool(n_classes >= min_classes and h_label >= min_label_entropy and mi_norm >= min_transport_mi_norm and acc >= min_transport_mi_norm)
    return EdgeQuotient(
        comp_edge=comp_edge,
        micro_edges=micro_edges,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_label=source_label,
        target_label=target_label,
        n_classes=int(n_classes),
        residual_bits=float(residual_bits),
        residual_qcoords=float(residual_qcoords),
        label_entropy_bits=float(h_label),
        transport_mi_bits=float(mi),
        transport_mi_norm=float(mi_norm),
        deterministic_accuracy=float(acc),
        live=live,
    )


# ---------------------------------------------------------------------------
# Directed cycle/path analysis
# ---------------------------------------------------------------------------
def directed_pair_edges(sys: OBG.FiniteRelationalSystem, comp_of: Dict[int, int]) -> Dict[CompEdge, List[Edge]]:
    out: Dict[CompEdge, List[Edge]] = defaultdict(list)
    for u, v in sys.edges():
        cu, cv = comp_of[u], comp_of[v]
        if cu != cv:
            out[(cu, cv)].append((u, v))
    return dict(out)


def orient_cycle_as_two_paths(cycle: Sequence[int], dir_edges: Set[CompEdge]) -> Optional[Tuple[int, int, List[List[int]]]]:
    """Return source,sink,two directed paths if a simple undirected cycle is a DAG diamond."""
    cyc = list(cycle)
    if len(cyc) < 3:
        return None
    # Edges available only between consecutive cycle vertices.
    adj: Dict[int, List[int]] = {v: [] for v in cyc}
    indeg = {v: 0 for v in cyc}
    outdeg = {v: 0 for v in cyc}
    ok_edges = 0
    for i in range(len(cyc)):
        a, b = cyc[i], cyc[(i + 1) % len(cyc)]
        if (a, b) in dir_edges and (b, a) not in dir_edges:
            adj[a].append(b); outdeg[a] += 1; indeg[b] += 1; ok_edges += 1
        elif (b, a) in dir_edges and (a, b) not in dir_edges:
            adj[b].append(a); outdeg[b] += 1; indeg[a] += 1; ok_edges += 1
        elif (a, b) in dir_edges and (b, a) in dir_edges:
            # Distinct SCCs cannot be mutually connected in a condensation DAG;
            # if this appears due a malformed input, skip for this audit.
            return None
        else:
            return None
    if ok_edges != len(cyc):
        return None
    sources = [v for v in cyc if indeg[v] == 0]
    sinks = [v for v in cyc if outdeg[v] == 0]
    if len(sources) != 1 or len(sinks) != 1:
        return None
    source, sink = sources[0], sinks[0]

    paths: List[List[int]] = []
    def dfs(v: int, path: List[int]):
        if v == sink:
            paths.append(path[:]); return
        for nb in adj.get(v, []):
            if nb in path:
                continue
            dfs(nb, path + [nb])
    dfs(source, [source])
    # A simple cycle diamond has exactly two directed simple paths around it.
    if len(paths) != 2:
        return None
    covered = set(paths[0]) | set(paths[1])
    if covered != set(cyc):
        return None
    return source, sink, paths


def best_deterministic_map_from_relation(pairs: Iterable[Tuple[int, int]], n_src: int, n_tgt: int) -> Tuple[Dict[int, int], float, bool, float]:
    counts = np.zeros((n_src, n_tgt), dtype=np.int64)
    for a, b in pairs:
        if 0 <= a < n_src and 0 <= b < n_tgt:
            counts[a, b] += 1
    total = int(counts.sum())
    if total == 0:
        return {}, 0.0, False, 0.0
    mapping: Dict[int, int] = {}
    covered_rows = 0
    correct = 0
    for a in range(n_src):
        row = counts[a]
        if row.sum() > 0:
            covered_rows += 1
            b = int(row.argmax())
            mapping[a] = b
            correct += int(row[b])
    acc = float(correct / total)
    # A bijection requires same alphabet size, every source row covered, and each target used exactly once.
    bij = bool(n_src == n_tgt and covered_rows == n_src and len(set(mapping.values())) == n_tgt)
    h_in = entropy_from_counts(counts.sum(axis=1))
    return mapping, acc, bij, h_in


def label_for_code(eq: EdgeQuotient, side: str, vals: Sequence[int]) -> Optional[int]:
    code = encode_tuple(vals, 4)  # overwritten below? kept for mypy-like clarity
    q = None
    # q is not stored in EdgeQuotient; infer impossible.  Caller should use helper below.
    return None


def _edge_side_code_label(eq: EdgeQuotient, side: str, assignment: Dict[int, int], q: int) -> Optional[int]:
    if side == "source":
        vals = [assignment.get(n, 0) for n in eq.source_nodes]
        code = encode_tuple(vals, q)
        return eq.source_label.get(code)
    if side == "target":
        vals = [assignment.get(n, 0) for n in eq.target_nodes]
        code = encode_tuple(vals, q)
        return eq.target_label.get(code)
    raise ValueError(side)


def local_label_map(
    sys: OBG.FiniteRelationalSystem,
    observer_nodes: Sequence[int],
    eq_a: EdgeQuotient,
    side_a: str,
    eq_b: EdgeQuotient,
    side_b: str,
    max_configs: int,
    rng: np.random.Generator,
) -> Tuple[Dict[int, int], float, bool, float]:
    """Relation between two edge-label readouts living on the same SCC observer."""
    q = sys.q
    nodes = tuple(sorted(set(int(x) for x in observer_nodes)))
    assignments = all_or_sample_assignments(q, nodes, max_configs, rng)
    pairs = []
    for vals in assignments:
        a = {node: val for node, val in zip(nodes, vals)}
        la = _edge_side_code_label(eq_a, side_a, a, q)
        lb = _edge_side_code_label(eq_b, side_b, a, q)
        if la is not None and lb is not None:
            pairs.append((int(la), int(lb)))
    n_src, n_tgt = eq_a.n_classes, eq_b.n_classes
    return best_deterministic_map_from_relation(pairs, n_src, n_tgt)


def compose_maps(f: Dict[int, int], g: Dict[int, int]) -> Dict[int, int]:
    """g after f: x -> g[f[x]]."""
    out = {}
    for a, b in f.items():
        if b in g:
            out[a] = g[b]
    return out


def identity_map(n: int) -> Dict[int, int]:
    return {i: i for i in range(n)}


def invert_permutation(m: Dict[int, int], n: int) -> Optional[Dict[int, int]]:
    if set(m.keys()) != set(range(n)) or set(m.values()) != set(range(n)):
        return None
    return {v: k for k, v in m.items()}


def permutation_cycle_type(perm: Dict[int, int], n: int) -> str:
    if n <= 0 or set(perm.keys()) != set(range(n)) or set(perm.values()) != set(range(n)):
        return "nonbijective"
    seen = set(); lens = []
    for i in range(n):
        if i in seen:
            continue
        cur = i; L = 0
        while cur not in seen:
            seen.add(cur); L += 1; cur = perm[cur]
        lens.append(L)
    lens = sorted(lens, reverse=True)
    if all(x == 1 for x in lens):
        return "identity"
    return "cycle_" + "_".join(str(x) for x in lens)


def generated_group_order(generators: List[Dict[int, int]], n: int, cap: int = 100000) -> Tuple[int, str]:
    gens = []
    for g in generators:
        if set(g.keys()) == set(range(n)) and set(g.values()) == set(range(n)):
            gens.append(tuple(g[i] for i in range(n)))
    if not gens:
        return 0, "none"
    ident = tuple(range(n))
    seen = {ident}
    dq = deque([ident])
    def comp(a, b):  # b after a
        return tuple(b[a[i]] for i in range(n))
    while dq and len(seen) < cap:
        cur = dq.popleft()
        for g in gens:
            for h in (comp(cur, g), comp(g, cur)):
                if h not in seen:
                    seen.add(h); dq.append(h)
    order = len(seen)
    if order == 1:
        name = "trivial"
    elif order == 2 and n >= 2:
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
    return order, name


def path_map(
    sys: OBG.FiniteRelationalSystem,
    comps: List[List[int]],
    edge_q: Dict[CompEdge, EdgeQuotient],
    path: Sequence[int],
    max_local_configs: int,
    rng: np.random.Generator,
) -> Tuple[Optional[Dict[int, int]], float, bool, int]:
    """Compose edge quotient identities and local observer maps along a directed path."""
    if len(path) < 2:
        return None, 0.0, False, 0
    first = (path[0], path[1])
    if first not in edge_q:
        return None, 0.0, False, 0
    n = edge_q[first].n_classes
    current = identity_map(n)  # source label of first edge -> target label of current edge
    accs = []
    bijs = []
    for i in range(1, len(path) - 1):
        prev_e = (path[i - 1], path[i])
        next_e = (path[i], path[i + 1])
        if prev_e not in edge_q or next_e not in edge_q:
            return None, 0.0, False, n
        eq_prev, eq_next = edge_q[prev_e], edge_q[next_e]
        lm, acc, bij, _h = local_label_map(sys, comps[path[i]], eq_prev, "target", eq_next, "source", max_local_configs, rng)
        accs.append(acc); bijs.append(bij)
        current = compose_maps(current, lm)
    # final edge identity already represented by same label ids of last edge; if path length 2, identity source->target.
    mean_acc = float(np.mean(accs)) if accs else 1.0
    return current, mean_acc, bool(all(bijs) if bijs else True), n


def analyze_cycle_holonomy(
    sys: OBG.FiniteRelationalSystem,
    comps: List[List[int]],
    edge_q: Dict[CompEdge, EdgeQuotient],
    cycle: Sequence[int],
    dir_edges: Set[CompEdge],
    max_local_configs: int,
    rng: np.random.Generator,
) -> Dict:
    oriented = orient_cycle_as_two_paths(cycle, dir_edges)
    row = dict(
        usable_diamond=0,
        valid_transport_cycle=0,
        valid_holonomy=0,
        nontrivial_holonomy=0,
        min_edge_classes=0,
        min_edge_residual_bits=0.0,
        min_edge_transport=0.0,
        local_map_accuracy=0.0,
        holonomy_type="not_usable",
        generated_group="none",
        group_order=0,
    )
    if oriented is None:
        return row
    source, sink, paths = oriented
    row.update(usable_diamond=1, source_observer=source, sink_observer=sink,
               path1=" ".join(map(str, paths[0])), path2=" ".join(map(str, paths[1])))
    directed_edges = [(p[i], p[i + 1]) for p in paths for i in range(len(p) - 1)]
    if not all(e in edge_q and edge_q[e].live for e in directed_edges):
        vals = [edge_q[e].n_classes for e in directed_edges if e in edge_q]
        row.update(min_edge_classes=int(min(vals) if vals else 0))
        return row
    m = min(edge_q[e].n_classes for e in directed_edges)
    # Require common alphabet size along all edges for a permutation holonomy.
    if any(edge_q[e].n_classes != m for e in directed_edges):
        row.update(min_edge_classes=int(m), holonomy_type="mixed_alphabet")
        return row

    f1, acc1, bij1, n1 = path_map(sys, comps, edge_q, paths[0], max_local_configs, rng)
    f2, acc2, bij2, n2 = path_map(sys, comps, edge_q, paths[1], max_local_configs, rng)
    row.update(min_edge_classes=int(m),
               min_edge_residual_bits=float(min(edge_q[e].residual_bits for e in directed_edges)),
               min_edge_transport=float(min(edge_q[e].transport_mi_norm for e in directed_edges)),
               local_map_accuracy=float(min(acc1, acc2)))
    if f1 is None or f2 is None or n1 != m or n2 != m:
        return row

    # Source and sink identifications between the two first/last edge labels.
    e1_first = (paths[0][0], paths[0][1]); e2_first = (paths[1][0], paths[1][1])
    e1_last = (paths[0][-2], paths[0][-1]); e2_last = (paths[1][-2], paths[1][-1])
    src_map, accs, bijs, _ = local_label_map(sys, comps[source], edge_q[e1_first], "source", edge_q[e2_first], "source", max_local_configs, rng)
    sink_map, accd, bijd, _ = local_label_map(sys, comps[sink], edge_q[e1_last], "target", edge_q[e2_last], "target", max_local_configs, rng)
    row["local_map_accuracy"] = float(min(row["local_map_accuracy"], accs, accd))
    if not (bij1 and bij2 and bijs and bijd):
        row.update(valid_transport_cycle=1, holonomy_type="nonbijective_local")
        return row

    # Compare path2 after source identification to path1 after sink identification.
    lhs = compose_maps(src_map, f2)       # S1 -> D2
    rhs = compose_maps(f1, sink_map)      # S1 -> D2
    inv_rhs = invert_permutation(rhs, m)
    if inv_rhs is None:
        row.update(valid_transport_cycle=1, holonomy_type="nonbijective_path")
        return row
    delta = compose_maps(lhs, inv_rhs)    # S1 -> S1
    if set(delta.keys()) != set(range(m)) or set(delta.values()) != set(range(m)):
        row.update(valid_transport_cycle=1, holonomy_type="nonbijective_delta")
        return row
    htype = permutation_cycle_type(delta, m)
    g_order, g_name = generated_group_order([delta], m)
    row.update(
        valid_transport_cycle=1,
        valid_holonomy=1,
        nontrivial_holonomy=int(htype != "identity"),
        holonomy_type=htype,
        group_order=int(g_order),
        generated_group=g_name,
        delta=" ".join(str(delta[i]) for i in range(m)),
    )
    return row


# ---------------------------------------------------------------------------
# System-level and experiment-level audit
# ---------------------------------------------------------------------------
def _component_map(comps: List[List[int]]) -> Dict[int, int]:
    return {v: ci for ci, comp in enumerate(comps) for v in comp}


def analyze_system_holonomy(
    sys: OBG.FiniteRelationalSystem,
    graph_id: int,
    capacity_threshold: float,
    cycle_capacity_threshold: float,
    max_channel_inputs: int,
    max_channel_backgrounds: int,
    max_local_configs: int,
    cycle_max_len: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict], List[Dict], Dict]:
    edges = sys.edges()
    comps = OBG.tarjan_scc(sys.k, edges)
    comp_of = _component_map(comps)
    pair_edges = directed_pair_edges(sys, comp_of)
    dir_edge_set = set(pair_edges)

    # Build observer-boundary cycles using the same geometry audit.
    _obs_rows, edge_rows, cycle_rows, geom_summary = OBG.analyze_system(
        sys, graph_id=graph_id, capacity_threshold=capacity_threshold,
        max_channel_inputs=max_channel_inputs, max_channel_backgrounds=max_channel_backgrounds,
        cycle_max_len=cycle_max_len, rng=rng,
    )

    # Extract exact common factors on every directed observer edge.
    edge_q: Dict[CompEdge, EdgeQuotient] = {}
    edge_out_rows: List[Dict] = []
    for ce, evs in pair_edges.items():
        eq = edge_common_factor(sys, ce, evs, max_channel_inputs, max_channel_backgrounds, rng)
        edge_q[ce] = eq
        edge_out_rows.append(dict(
            graph_id=graph_id,
            source_observer=ce[0], target_observer=ce[1],
            source_nodes=" ".join(map(str, eq.source_nodes)),
            target_nodes=" ".join(map(str, eq.target_nodes)),
            n_micro_edges=len(eq.micro_edges),
            shared_classes=eq.n_classes,
            residual_bits=eq.residual_bits,
            residual_qcoords=eq.residual_qcoords,
            label_entropy_bits=eq.label_entropy_bits,
            transport_mi_norm=eq.transport_mi_norm,
            deterministic_accuracy=eq.deterministic_accuracy,
            live_quotient=int(eq.live),
        ))

    hol_rows: List[Dict] = []
    for cr in cycle_rows:
        # Only analyze cycles that are live at observer-boundary capacity threshold.
        if float(cr.get("min_edge_mi_bits", 0.0)) < cycle_capacity_threshold:
            continue
        cyc = tuple(int(x) for x in str(cr["observers"]).split())
        hr = analyze_cycle_holonomy(sys, comps, edge_q, cyc, dir_edge_set, max_local_configs, rng)
        hr.update(
            graph_id=graph_id,
            cycle_id=int(cr["cycle_id"]),
            length=int(cr["length"]),
            observers=cr["observers"],
            min_observer_edge_mi_bits=float(cr["min_edge_mi_bits"]),
            mean_observer_edge_mi_bits=float(cr["mean_edge_mi_bits"]),
        )
        hol_rows.append(hr)

    usable = sum(int(r.get("usable_diamond", 0)) for r in hol_rows)
    valid = sum(int(r.get("valid_holonomy", 0)) for r in hol_rows)
    nontriv = sum(int(r.get("nontrivial_holonomy", 0)) for r in hol_rows)
    live_edges = sum(int(r.get("live_quotient", 0)) for r in edge_out_rows)
    summary = dict(
        graph_id=graph_id,
        n_scc=geom_summary["n_scc"],
        n_observer_cycles=geom_summary["n_observer_cycles"],
        n_live_observer_cycles=geom_summary["n_live_observer_cycles"],
        n_tested_cycles=len(hol_rows),
        n_usable_diamonds=usable,
        n_valid_holonomy=valid,
        n_nontrivial_holonomy=nontriv,
        n_observer_edges=len(edge_out_rows),
        n_live_quotient_edges=live_edges,
        full_boundary_violation=geom_summary.get("full_boundary_violation", 0),
    )
    return edge_out_rows, hol_rows, summary


def run_audit(
    q: int,
    graphs: int,
    vertices: int,
    graph_mode: str,
    components: int,
    edge_prob: float,
    inter_prob: float,
    extra_intra_prob: float,
    max_pred: Optional[int],
    capacity_threshold: float,
    cycle_capacity_threshold: float,
    max_channel_inputs: int,
    max_channel_backgrounds: int,
    max_local_configs: int,
    cycle_max_len: int,
    seed: int,
    verbose: bool = True,
) -> Tuple[object, object, Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for run_audit/CLI output")
    rng0 = np.random.default_rng(seed)
    all_edges: List[Dict] = []
    all_hol: List[Dict] = []
    summaries: List[Dict] = []
    for gi in range(graphs):
        grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
        if graph_mode == "er":
            sys = OBG.make_random_system(q, vertices, edge_prob, grng, max_pred=max_pred)
        elif graph_mode == "componented":
            sys = OBG.make_componented_system(q, vertices, components, inter_prob, extra_intra_prob, grng, max_pred=max_pred)
        else:
            raise ValueError(f"unknown graph mode: {graph_mode}")
        erows, hrows, summ = analyze_system_holonomy(
            sys, gi, capacity_threshold, cycle_capacity_threshold,
            max_channel_inputs, max_channel_backgrounds, max_local_configs,
            cycle_max_len, grng,
        )
        all_edges.extend(erows); all_hol.extend(hrows); summaries.append(summ)
        if verbose:
            print(f"observer-holonomy graph={gi} tested={summ['n_tested_cycles']} usable={summ['n_usable_diamonds']} valid={summ['n_valid_holonomy']} nontriv={summ['n_nontrivial_holonomy']}", flush=True)

    edge_df = pd.DataFrame(all_edges)
    hol_df = pd.DataFrame(all_hol)
    summ_df = pd.DataFrame(summaries)
    n_tested = int(summ_df["n_tested_cycles"].sum()) if len(summ_df) else 0
    n_usable = int(summ_df["n_usable_diamonds"].sum()) if len(summ_df) else 0
    n_valid = int(summ_df["n_valid_holonomy"].sum()) if len(summ_df) else 0
    n_nontriv = int(summ_df["n_nontrivial_holonomy"].sum()) if len(summ_df) else 0
    n_live_edges = int(summ_df["n_live_quotient_edges"].sum()) if len(summ_df) else 0

    if n_nontriv > 0:
        verdict = "OBSERVER-RELATIVE HOLONOMY SIGNAL: discovered observer cycles carry nontrivial gauge-covariant holonomy candidates"
    elif n_valid > 0:
        verdict = "OBSERVER-RELATIVE TRANSPORT CYCLE SIGNAL: discovered cycles carry valid transported quotients but holonomy is flat"
    elif n_usable > 0:
        verdict = "OBSERVER-RELATIVE DIAMOND GEOMETRY ONLY: live cycles orient as two-path diamonds but transported quotient holonomy was not found"
    elif n_tested > 0 or n_live_edges > 0:
        verdict = "OBSERVER-RELATIVE QUOTIENT EDGE SIGNAL: live quotient edges appear, but no usable cycle holonomy was found"
    else:
        verdict = "NO OBSERVER-RELATIVE HOLONOMY SIGNAL in this regime"

    group_counts = {}
    hol_type_counts = {}
    if len(hol_df) and "valid_holonomy" in hol_df:
        valid_h = hol_df[hol_df["valid_holonomy"] == 1]
        if len(valid_h):
            group_counts = {str(k): int(v) for k, v in Counter(valid_h["generated_group"].astype(str)).items()}
            hol_type_counts = {str(k): int(v) for k, v in Counter(valid_h["holonomy_type"].astype(str)).items()}

    summary = dict(
        verdict=verdict,
        q=q,
        graph_mode=graph_mode,
        n_graphs=int(graphs),
        n_vertices=int(vertices),
        total_tested_cycles=n_tested,
        total_usable_diamonds=n_usable,
        total_valid_holonomy=n_valid,
        total_nontrivial_holonomy=n_nontriv,
        live_edge_quotient_fraction=float(edge_df["live_quotient"].mean()) if len(edge_df) and "live_quotient" in edge_df else 0.0,
        graph_fraction_with_valid_holonomy=float((summ_df["n_valid_holonomy"] > 0).mean()) if len(summ_df) else 0.0,
        graph_fraction_with_nontrivial_holonomy=float((summ_df["n_nontrivial_holonomy"] > 0).mean()) if len(summ_df) else 0.0,
        usable_diamond_fraction=float(n_usable / n_tested) if n_tested else 0.0,
        valid_holonomy_fraction=float(n_valid / n_tested) if n_tested else 0.0,
        nontrivial_holonomy_fraction=float(n_nontriv / n_valid) if n_valid else 0.0,
        mean_edge_shared_classes=float(edge_df["shared_classes"].mean()) if len(edge_df) and "shared_classes" in edge_df else 0.0,
        mean_edge_residual_bits=float(edge_df["residual_bits"].mean()) if len(edge_df) and "residual_bits" in edge_df else 0.0,
        mean_edge_transport=float(edge_df["transport_mi_norm"].mean()) if len(edge_df) and "transport_mi_norm" in edge_df else 0.0,
        group_counts=group_counts,
        holonomy_type_counts=hol_type_counts,
        full_boundary_violations=int(summ_df["full_boundary_violation"].sum()) if len(summ_df) else 0,
    )
    return edge_df, hol_df, summary


# ---------------------------------------------------------------------------
# Output/plot/CLI
# ---------------------------------------------------------------------------
def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def write_outputs(edge_df, hol_df, summary: Dict, out: str) -> None:
    edge_df.to_csv(out, index=False)
    hol_df.to_csv(derived_path(out, "_cycles"), index=False)
    with open(derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def make_plot(edge_df, hol_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(summary.get("verdict", "observer-relative holonomy"), fontsize=12)

    ax = axes[0]
    if len(edge_df) and "residual_bits" in edge_df:
        ax.hist(edge_df["residual_bits"].dropna().values, bins=20)
    ax.set_title("edge quotient residual")
    ax.set_xlabel("bits")
    ax.set_ylabel("count")

    ax = axes[1]
    if len(hol_df) and "min_observer_edge_mi_bits" in hol_df:
        ax.hist(hol_df["min_observer_edge_mi_bits"].dropna().values, bins=20)
    ax.set_title("tested cycle min edge MI")
    ax.set_xlabel("bits")

    ax = axes[2]
    if len(hol_df) and "holonomy_type" in hol_df:
        vc = hol_df[hol_df.get("valid_holonomy", 0) == 1]["holonomy_type"].astype(str).value_counts()
        if len(vc):
            ax.bar(range(len(vc)), vc.values)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(vc.index, rotation=35, ha="right")
    ax.set_title("valid holonomy types")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Observer-relative holonomy audit on discovered SCC-boundary cycles.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--graphs", type=int, default=100)
    p.add_argument("--vertices", type=int, default=18)
    p.add_argument("--graph-mode", choices=["componented", "er"], default="componented")
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.16)
    p.add_argument("--inter-prob", type=float, default=0.35)
    p.add_argument("--extra-intra-prob", type=float, default=0.25)
    p.add_argument("--max-pred", type=int, default=4)
    p.add_argument("--capacity-threshold", type=float, default=0.05)
    p.add_argument("--cycle-capacity-threshold", type=float, default=0.05)
    p.add_argument("--max-channel-inputs", type=int, default=256)
    p.add_argument("--max-channel-backgrounds", type=int, default=256)
    p.add_argument("--max-local-configs", type=int, default=512)
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/observer_relative_holonomy.csv")
    p.add_argument("--plot", default="example_results/fig_observer_relative_holonomy.png")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    edge_df, hol_df, summary = run_audit(
        q=args.q,
        graphs=args.graphs,
        vertices=args.vertices,
        graph_mode=args.graph_mode,
        components=args.components,
        edge_prob=args.edge_prob,
        inter_prob=args.inter_prob,
        extra_intra_prob=args.extra_intra_prob,
        max_pred=args.max_pred,
        capacity_threshold=args.capacity_threshold,
        cycle_capacity_threshold=args.cycle_capacity_threshold,
        max_channel_inputs=args.max_channel_inputs,
        max_channel_backgrounds=args.max_channel_backgrounds,
        max_local_configs=args.max_local_configs,
        cycle_max_len=args.cycle_max_len,
        seed=args.seed,
        verbose=not args.quiet,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_outputs(edge_df, hol_df, summary, args.out)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plot(edge_df, hol_df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
