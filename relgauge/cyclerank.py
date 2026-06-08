"""
cyclerank.py -- how many coordinates does the schedule-gauge remove?

Define the gauge deficit of a system as
    deficit = |V| - log_q |reachable-in-one-step under all admissible schedules|
in the q -> infinity trend (so the (1-1/e) non-injectivity drops out of log_q).

Two hypotheses:
  H_betti    : deficit = first Betti number beta_1 = |E| - |V| + (#components)
  H_observer : deficit = number of SCCs with >= 2 vertices  (one global
               'update phase' per genuine observer)

The in-tree argument (README) PROVES deficit = 1 for any single SCC of >= 2
vertices regardless of internal density, refuting H_betti and supporting
H_observer.  This module measures it and reports which hypothesis the data back.

Multi-SCC systems are supported via admissible (condensation-respecting)
schedules so the additivity 'deficit = #observers' can be checked directly.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import core as C


# --------------------------------------------------------------------------- #
# graph invariants
# --------------------------------------------------------------------------- #
def edge_list(sys: C.RelationalSystem):
    return [(p, v) for v in range(sys.k) for p in sys.preds[v]]


def betti1(sys: C.RelationalSystem) -> int:
    E = len(edge_list(sys))
    V = sys.k
    adj = {v: set() for v in range(V)}
    for (u, v) in edge_list(sys):
        adj[u].add(v); adj[v].add(u)            # undirected components
    # count weakly-connected components
    seen = set(); comps = 0
    for s in range(V):
        if s in seen:
            continue
        comps += 1; stack = [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); stack.extend(adj[x] - seen)
    return E - V + comps


def n_observers(sys: C.RelationalSystem) -> int:
    """Number of SCCs with >= 2 vertices (genuine feedback observers)."""
    adj = {v: set() for v in range(sys.k)}
    for (u, v) in edge_list(sys):
        adj[u].add(v)
    comps = C.tarjan_scc(adj, range(sys.k))
    return sum(1 for c in comps if len(c) >= 2)


# --------------------------------------------------------------------------- #
# admissible (condensation-respecting) schedules for multi-SCC graphs
# --------------------------------------------------------------------------- #
def admissible_schedules(sys: C.RelationalSystem, cap=5040):
    """All vertex orderings that respect the condensation partial order
    (free within each SCC, SCC-order topological).  Falls back to a sample if
    there are too many."""
    adj = {v: set() for v in range(sys.k)}
    for (u, v) in edge_list(sys):
        adj[u].add(v)
    comps = C.tarjan_scc(adj, range(sys.k))
    comp_of = {}
    for i, c in enumerate(comps):
        for v in c:
            comp_of[v] = i
    # condensation edges
    cadj = {i: set() for i in range(len(comps))}
    indeg = {i: 0 for i in range(len(comps))}
    for (u, v) in edge_list(sys):
        if comp_of[u] != comp_of[v] and comp_of[v] not in cadj[comp_of[u]]:
            cadj[comp_of[u]].add(comp_of[v])
    # all permutations of V that are consistent: simplest correct approach for
    # small k -- filter all k! perms by the partial order (vertex u before v if
    # comp(u) strictly precedes comp(v) in condensation reachability).
    # precompute strict precedence between comps (transitive closure)
    reach = {i: set() for i in range(len(comps))}
    for i in range(len(comps)):
        stack = list(cadj[i]); 
        while stack:
            j = stack.pop()
            if j not in reach[i]:
                reach[i].add(j); stack.extend(cadj[j])
    must_before = {}   # (u,v): u must come before v
    out = []
    perms = itertools.permutations(range(sys.k))
    for p in perms:
        pos = {v: i for i, v in enumerate(p)}
        ok = True
        for u in range(sys.k):
            for v in range(sys.k):
                if comp_of[v] in reach[comp_of[u]] and pos[u] > pos[v]:
                    ok = False; break
            if not ok:
                break
        if ok:
            out.append(p)
            if len(out) >= cap:
                break
    return out


def step_maps_admissible(sys: C.RelationalSystem, cap=5040) -> np.ndarray:
    scheds = admissible_schedules(sys, cap=cap)
    out = np.empty((len(scheds), sys.q ** sys.k), dtype=np.int64)
    for s, sched in enumerate(scheds):
        out[s] = sys.step_map(sched)
    return out


# --------------------------------------------------------------------------- #
# deficit measurement
# --------------------------------------------------------------------------- #
def gauge_deficit(sys: C.RelationalSystem, multi_scc=False) -> float:
    if multi_scc:
        sm = step_maps_admissible(sys)
    else:
        sm = sys.all_step_maps()
    img = C.image_states(sm)
    dim_coords = np.log(len(img)) / np.log(sys.q)     # in units of q-ary coords
    return sys.k - dim_coords


# --------------------------------------------------------------------------- #
# single-SCC sweep: deficit vs density (beta_1) and q  -- refutes H_betti
# --------------------------------------------------------------------------- #
def run_single_scc(k=4, qs=(4, 8, 16, 32), extras=(0, 1, 2, 3),
                   n_instances=60, base_seed=0, verbose=True) -> pd.DataFrame:
    rows = []
    for q in qs:
        for extra in extras:
            for inst in range(n_instances):
                seed = (base_seed * 6151 + q * 31 + extra * 7 + inst) % 2**32
                rng = np.random.default_rng(seed)
                sysm = C.make_random_scc(k, q, extra, rng)
                rows.append(dict(k=k, q=q, extra=extra,
                                 beta1=betti1(sysm), n_obs=n_observers(sysm),
                                 deficit=gauge_deficit(sysm), seed=seed))
        if verbose:
            print(f"  single-SCC q={q} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# multi-SCC sweep: deficit additivity = #observers
# --------------------------------------------------------------------------- #
def make_chain_of_cycles(scc_sizes, q, rng) -> C.RelationalSystem:
    """Build a DAG of cycles: each cycle is an SCC; consecutive cycles linked by
    a single feed-forward edge.  #observers = #(cycles with size>=2)."""
    preds = []
    offset = 0
    starts = []
    k_total = sum(scc_sizes)
    # first build cycle backbones
    pred_sets = [set() for _ in range(k_total)]
    for sz in scc_sizes:
        starts.append(offset)
        if sz == 1:
            pass  # isolated vertex (trivial SCC) -- gets a feed-forward input below
        else:
            for j in range(sz):
                u = offset + ((j - 1) % sz)
                v = offset + j
                pred_sets[v].add(u)
        offset += sz
    # feed-forward links between consecutive SCCs (DAG, no back edges)
    for i in range(1, len(scc_sizes)):
        src = starts[i - 1]                  # a vertex in previous SCC
        dst = starts[i]                      # a vertex in this SCC
        pred_sets[dst].add(src)
    preds = [tuple(sorted(pred_sets[v])) for v in range(k_total)]
    # ensure no vertex has indeg 0 except the very first source chain start
    rules = []
    for v in range(k_total):
        d = len(preds[v])
        rules.append(rng.integers(0, q, size=q ** max(d, 0), dtype=np.int64)
                     if d > 0 else np.array([0], dtype=np.int64))
    # represent indeg-0 vertices as reading nothing (identity-ish via const);
    # core treats empty preds as identity, fine for source.
    sys = C.RelationalSystem(k_total, q, preds, rules)
    return sys


def run_multi_scc(configs, q=4, n_instances=40, base_seed=0,
                  verbose=True) -> pd.DataFrame:
    """configs: list of scc_size tuples, e.g. [(2,), (2,2), (3,2), (2,2,2)]."""
    rows = []
    for ci, sizes in enumerate(configs):
        if sum(sizes) > 7:                    # keep q^k tractable
            continue
        for inst in range(n_instances):
            seed = (base_seed * 977 + ci * 41 + inst) % 2**32
            rng = np.random.default_rng(seed)
            sysm = make_chain_of_cycles(sizes, q, rng)
            rows.append(dict(config=str(sizes), k=sysm.k,
                             beta1=betti1(sysm), n_obs=n_observers(sysm),
                             deficit=gauge_deficit(sysm, multi_scc=True),
                             seed=seed))
        if verbose:
            print(f"  multi-SCC {sizes} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyze(df_single: pd.DataFrame, df_multi: pd.DataFrame | None = None) -> dict:
    res = dict()
    # single-SCC: deficit should -> 1 as q grows, independent of beta1/extra
    qmax = df_single.q.max()
    big = df_single[df_single.q == qmax]
    res["single_scc"] = dict(
        qmax=int(qmax),
        deficit_by_extra_at_qmax={
            int(e): float(big[big.extra == e].deficit.mean())
            for e in sorted(big.extra.unique())},
        beta1_by_extra={
            int(e): float(df_single[df_single.extra == e].beta1.mean())
            for e in sorted(df_single.extra.unique())},
        deficit_trend_in_q={
            int(q): float(df_single[df_single.q == q].deficit.mean())
            for q in sorted(df_single.q.unique())})
    # verdict: is deficit ~ 1 regardless of beta1?
    defs = list(res["single_scc"]["deficit_by_extra_at_qmax"].values())
    betas = list(res["single_scc"]["beta1_by_extra"].values())
    flat_at_1 = (max(defs) - min(defs) < 0.4) and abs(np.mean(defs) - 1.0) < 0.5
    tracks_beta = np.corrcoef(defs, betas)[0, 1] > 0.9 and (max(betas) - min(betas) > 1)
    if flat_at_1 and not tracks_beta:
        res["single_verdict"] = ("OBSERVER-COUNT confirmed: deficit ~ 1 for any "
                                 "single SCC, independent of beta_1 (H_betti "
                                 "refuted)")
    elif tracks_beta:
        res["single_verdict"] = "deficit tracks beta_1 (H_betti supported)"
    else:
        res["single_verdict"] = "INCONCLUSIVE (need larger q)"
    if df_multi is not None and len(df_multi):
        res["multi_scc"] = []
        ok = 0; tot = 0
        for cfg in sorted(df_multi.config.unique()):
            s = df_multi[df_multi.config == cfg]
            d = float(s.deficit.mean()); no = int(s.n_obs.iloc[0]); b1 = int(s.beta1.iloc[0])
            res["multi_scc"].append(dict(config=cfg, deficit=d, n_obs=no, beta1=b1))
            tot += 1
            if abs(d - no) < 0.5:
                ok += 1
        res["multi_verdict"] = (f"deficit = #observers in {ok}/{tot} configs "
                                f"(additivity of one phase per observer)")
    return res


if __name__ == "__main__":
    print("single-SCC: deficit vs density (refutes beta_1) ...")
    ds = run_single_scc(k=4, qs=(4, 8, 16, 32), extras=(0, 1, 2, 3),
                        n_instances=60, verbose=False)
    print("multi-SCC: additivity = #observers ...")
    dm = run_multi_scc([(2,), (3,), (2, 2), (3, 2), (2, 2, 2)], q=4,
                       n_instances=40, verbose=False)
    res = analyze(ds, dm)
    import json
    print(json.dumps(res, indent=2, default=float))
