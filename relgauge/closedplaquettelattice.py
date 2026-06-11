"""
closedplaquettelattice.py -- closed SCC plaquette-lattice audit/search.

This module is the first non-feed-forward Step-3 test.  Earlier shared-square
and micro-lattice modules use a source-to-sink transporter topology.  Here the
microscopic graph is a periodic link lattice: every link variable is part of one
strongly connected component.  There is no source, no sink, and no topological
sweep that overwrites hidden inputs.

The module asks whether a nontrivial binary transported quotient can survive in
a recurrent closed medium.  It keeps the search criterion at the quotient level:
find a nontrivial binary link quotient whose one-step quotient dynamics is
schedule-consistent and deterministic.  Plaquette flux entropy, flux cycles, and
recurrent flux quotients are reported post-hoc diagnostics, not fitness terms.

Recommended controls
--------------------
python -m relgauge.closedplaquettelattice 4 \
  --mode sweep --sizes 2x2 --ensembles c2xor,copy,random --instances 50 \
  --n-initial 4096 --horizon 24 --schedule-samples 8 \
  --out example_results/closed_plaquette_sweep_q4.csv \
  --plot example_results/fig_closed_plaquette_sweep_q4.png

Blind closed-SCC selection
--------------------------
python -m relgauge.closedplaquettelattice 4 \
  --mode blind --sizes 2x2 --null-samples 100 --runs 5 --population 24 \
  --generations 100 --n-initial 4096 --horizon 24 --schedule-samples 8 \
  --out example_results/closed_plaquette_blind_q4_2x2.csv \
  --plot example_results/fig_closed_plaquette_blind_q4_2x2.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import core as C

Ensemble = Literal["random", "copy", "c2copy", "c2xor", "permutive"]


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _parse_size(s: str) -> tuple[int, int]:
    if "x" in str(s).lower():
        a, b = str(s).lower().split("x", 1)
        return int(a), int(b)
    n = int(s)
    return n, n


def _parse_list(s: str, typ=str):
    if not s:
        return []
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    tot = float(counts.sum())
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def _entropy_discrete(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    _, c = np.unique(x, return_counts=True, axis=0 if x.ndim > 1 else None)
    return _entropy_from_counts(c)


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return 0.0
    # Convert rows to stable integer labels.
    _, xi = np.unique(x, return_inverse=True, axis=0 if x.ndim > 1 else None)
    _, yi = np.unique(y, return_inverse=True, axis=0 if y.ndim > 1 else None)
    nx = int(xi.max()) + 1 if len(xi) else 0
    ny = int(yi.max()) + 1 if len(yi) else 0
    if nx == 0 or ny == 0:
        return 0.0
    joint = np.zeros((nx, ny), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)
    return float(_entropy_from_counts(joint.sum(axis=1)) + _entropy_from_counts(joint.sum(axis=0)) - _entropy_from_counts(joint.reshape(-1)))


def _random_rule(q: int, indeg: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, q, size=int(q) ** int(indeg), dtype=np.int64)


def _all_inputs(indeg: int, q: int) -> np.ndarray:
    return C.all_states(int(indeg), int(q)) if indeg else np.zeros((1, 0), dtype=np.int64)


def _rule_copy_input(q: int, indeg: int, pos: int, transform=None) -> np.ndarray:
    X = _all_inputs(indeg, q)
    y = X[:, int(pos)].astype(np.int64)
    if transform is not None:
        y = transform(y)
    return np.asarray(y, dtype=np.int64) % int(q)


def _bit_partition_values(q: int, mask: int) -> np.ndarray:
    """Return length-q 0/1 labels for a nontrivial alphabet partition.

    mask and complement are identified; callers should enumerate only masks
    with value 0 in class 0 to avoid duplicates.
    """
    q = int(q); mask = int(mask)
    lab = np.zeros(q, dtype=np.int64)
    for a in range(q):
        if (mask >> a) & 1:
            lab[a] = 1
    return lab


def binary_partitions(q: int) -> list[np.ndarray]:
    """All nontrivial binary partitions of [q], modulo label complement."""
    q = int(q)
    out = []
    for mask in range(1, (1 << q) - 1):
        # fix symbol 0 in class 0 to quotient label complement.
        if (mask & 1) != 0:
            continue
        lab = _bit_partition_values(q, mask)
        if 0 < int(lab.sum()) < q:
            out.append(lab)
    return out


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


def _random_schedules(k: int, n: int, rng: np.random.Generator, include_canonical: tuple[int, ...] | None = None) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    if include_canonical is not None:
        out.append(tuple(int(x) for x in include_canonical))
    seen = set(out)
    for _ in range(max(0, int(n) - len(out))):
        p = tuple(int(x) for x in rng.permutation(int(k)).tolist())
        if p not in seen:
            out.append(p); seen.add(p)
    return out


def _determinism_accuracy(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Best deterministic map accuracy y=f(x), treating rows as states."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return 0.0, 0
    _, xi = np.unique(x, return_inverse=True, axis=0 if x.ndim > 1 else None)
    _, yi = np.unique(y, return_inverse=True, axis=0 if y.ndim > 1 else None)
    total = 0
    deterministic = True
    for a in range(int(xi.max()) + 1 if len(xi) else 0):
        vals, counts = np.unique(yi[xi == a], return_counts=True)
        if len(vals) > 1:
            deterministic = False
        total += int(counts.max()) if len(counts) else 0
    return float(total / len(xi)), int(deterministic)


# --------------------------------------------------------------------------- #
# Closed periodic link lattice
# --------------------------------------------------------------------------- #
@dataclass
class ClosedPlaquetteLattice:
    sys: C.RelationalSystem
    q: int
    rows: int
    cols: int
    H: np.ndarray  # (rows, cols) horizontal link vertices, from (r,c)->(r,c+1)
    V: np.ndarray  # (rows, cols) vertical link vertices, from (r,c)->(r+1,c)
    canonical_schedule: tuple[int, ...]
    meta: dict

    @property
    def n_links(self) -> int:
        return int(self.sys.k)

    @property
    def n_plaquettes(self) -> int:
        return int(self.rows * self.cols)


def _edge_neighborhood(rows: int, cols: int, H: np.ndarray, V: np.ndarray, kind: str, r: int, c: int) -> list[int]:
    """Adjacent links sharing endpoints/plaquettes with a given periodic link."""
    R, Cc = int(rows), int(cols)
    r = int(r) % R; c = int(c) % Cc
    if kind == "H":
        # self, horizontal left/right, verticals at both endpoints above/below.
        return [
            int(H[r, c]),
            int(H[r, (c - 1) % Cc]), int(H[r, (c + 1) % Cc]),
            int(V[r, c]), int(V[(r - 1) % R, c]),
            int(V[r, (c + 1) % Cc]), int(V[(r - 1) % R, (c + 1) % Cc]),
        ]
    else:
        return [
            int(V[r, c]),
            int(V[(r - 1) % R, c]), int(V[(r + 1) % R, c]),
            int(H[r, c]), int(H[r, (c - 1) % Cc]),
            int(H[(r + 1) % R, c]), int(H[(r + 1) % R, (c - 1) % Cc]),
        ]


def make_closed_lattice(
    q: int,
    rows: int,
    cols: int,
    rng: np.random.Generator,
    ensemble: Ensemble = "random",
    mutation_rate: float = 0.0,
    xor_bias: float = 0.5,
) -> ClosedPlaquetteLattice:
    """Construct a periodic link lattice.  The predecessor graph is one SCC.

    There are rows*cols horizontal and rows*cols vertical link variables.  Each
    link reads adjacent links in the periodic cell complex, so there is no
    source/sink boundary.  Control ensembles install known binary quotient
    dynamics; random is a null.
    """
    q = int(q); rows = int(rows); cols = int(cols)
    if rows < 2 or cols < 2:
        raise ValueError("closed plaquette lattice requires at least 2x2")
    n = 2 * rows * cols
    H = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    V = (rows * cols + np.arange(rows * cols, dtype=np.int64)).reshape(rows, cols)
    pred_sets: list[set[int]] = [set() for _ in range(n)]
    # Periodic adjacent-link predecessor graph.
    for r in range(rows):
        for c in range(cols):
            for v in _edge_neighborhood(rows, cols, H, V, "H", r, c):
                pred_sets[int(H[r, c])].add(int(v))
            for v in _edge_neighborhood(rows, cols, H, V, "V", r, c):
                pred_sets[int(V[r, c])].add(int(v))
    preds = [tuple(sorted(s)) for s in pred_sets]
    rules = [_random_rule(q, len(preds[v]), rng) for v in range(n)]
    sys = C.RelationalSystem(n, q, preds, rules, meta=dict(ensemble="closed_plaquette", rows=rows, cols=cols))

    # Helpers for control rules.  Binary label is value % 2.  Output values are
    # kept in {0,1}, so any q>=2 alphabet carries a C2 quotient but not more.
    def make_bit_rule(v: int, mode: str) -> np.ndarray:
        pred = preds[int(v)]
        X = _all_inputs(len(pred), q)
        bits = X % 2
        if mode == "copy":
            # Copy self bit.
            pos = pred.index(int(v))
            y = bits[:, pos]
        elif mode == "xor":
            # XOR a small deterministic subset including self and one/two
            # adjacent links.  This gives recurrent closed dynamics rather than
            # a feed-forward projection.
            take = min(3, bits.shape[1])
            y = np.bitwise_xor.reduce(bits[:, :take], axis=1)
        else:
            y = np.zeros(X.shape[0], dtype=np.int64)
        return y.astype(np.int64) % q

    if ensemble == "copy":
        for v in range(n):
            sys.rules[v] = _rule_copy_input(q, len(preds[v]), preds[v].index(v))
    elif ensemble == "c2copy":
        for v in range(n):
            sys.rules[v] = make_bit_rule(v, "copy")
    elif ensemble == "c2xor":
        for v in range(n):
            sys.rules[v] = make_bit_rule(v, "xor")
    elif ensemble == "permutive":
        for v in range(n):
            pred = preds[v]
            pos = pred.index(v)
            perm = rng.permutation(q).astype(np.int64)
            sys.rules[v] = _rule_copy_input(q, len(pred), pos, lambda x, p=perm: p[np.asarray(x, dtype=np.int64)])
    elif ensemble == "random":
        pass
    else:
        raise ValueError(f"unknown ensemble {ensemble!r}")

    if mutation_rate > 0:
        for v in range(n):
            rule = sys.rules[v].copy()
            mask = rng.random(rule.size) < float(mutation_rate)
            if np.any(mask):
                jumps = rng.integers(1, q, size=int(mask.sum()), dtype=np.int64)
                rule[mask] = (rule[mask] + jumps) % q
            sys.rules[v] = rule

    canonical = tuple(range(n))
    adj = {v: set(preds[v]) for v in range(n)}
    # Edges in pred graph are pred -> v; tarjan wants outgoing.  Build outgoing.
    out_adj = {v: set() for v in range(n)}
    for v, ps in enumerate(preds):
        for p in ps:
            out_adj[int(p)].add(int(v))
    comps = C.tarjan_scc(out_adj, range(n))
    meta = dict(rule_ensemble=ensemble, mutation_rate=float(mutation_rate), closed_scc=bool(len(comps) == 1), n_scc=int(len(comps)))
    return ClosedPlaquetteLattice(sys=sys, q=q, rows=rows, cols=cols, H=H, V=V, canonical_schedule=canonical, meta=meta)


# --------------------------------------------------------------------------- #
# Readouts and metrics
# --------------------------------------------------------------------------- #
def _initial_states(lat: ClosedPlaquetteLattice, n_initial: int, rng: np.random.Generator, mode: str = "joint_random") -> np.ndarray:
    q = lat.q; k = lat.n_links
    if mode == "all" and q ** k <= int(n_initial):
        return C.all_states(k, q)
    return rng.integers(0, q, size=(int(n_initial), k), dtype=np.int64)


def _bits_from_states(S: np.ndarray, partition: np.ndarray) -> np.ndarray:
    return np.asarray(partition, dtype=np.int64)[np.asarray(S, dtype=np.int64)]


def plaquette_flux_bits(lat: ClosedPlaquetteLattice, B: np.ndarray) -> np.ndarray:
    """Z2 plaquette flux vector for each row of bit-state B."""
    R, Cc = lat.rows, lat.cols
    out = np.zeros((B.shape[0], R * Cc), dtype=np.int64)
    idx = 0
    for r in range(R):
        for c in range(Cc):
            h_top = int(lat.H[r, c])
            v_right = int(lat.V[r, (c + 1) % Cc])
            h_bottom = int(lat.H[(r + 1) % R, c])
            v_left = int(lat.V[r, c])
            out[:, idx] = B[:, h_top] ^ B[:, v_right] ^ B[:, h_bottom] ^ B[:, v_left]
            idx += 1
    return out


def _flux_vector_to_int(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.int64)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    # For up to ~32 plaquettes this is safe in int64.
    powers = (1 << np.arange(F.shape[1], dtype=np.int64))
    return (F * powers).sum(axis=1).astype(np.int64)


def _transition_metrics(lat: ClosedPlaquetteLattice, S0: np.ndarray, partition: np.ndarray, schedules: list[tuple[int, ...]], horizon: int, rng: np.random.Generator) -> dict:
    q = lat.q
    # One-step schedule consistency and quotient determinism.
    B0 = _bits_from_states(S0, partition)
    succ_bits = []
    succ_flux = []
    for sched in schedules:
        S1 = _step_states(lat.sys, S0, sched)
        B1 = _bits_from_states(S1, partition)
        succ_bits.append(B1)
        succ_flux.append(plaquette_flux_bits(lat, B1))
    B1c = succ_bits[0]
    F1c = succ_flux[0]
    # Strict schedule agreement on flux and link quotient.
    if len(schedules) <= 1:
        link_sched = 1.0; flux_sched = 1.0
    else:
        link_agree = np.ones(len(S0), dtype=bool)
        flux_agree = np.ones(len(S0), dtype=bool)
        for j in range(1, len(schedules)):
            link_agree &= np.all(succ_bits[j] == B1c, axis=1)
            flux_agree &= np.all(succ_flux[j] == F1c, axis=1)
        link_sched = float(link_agree.mean())
        flux_sched = float(flux_agree.mean())
    det_acc, det_exact = _determinism_accuracy(B0, B1c)
    H_B = _entropy_discrete(B0)
    H_F1 = _entropy_discrete(F1c)
    MI_B = _mutual_info_discrete(B0, B1c)
    flux = plaquette_flux_bits(lat, B0)
    nontriv = float(flux.mean()) if flux.size else 0.0
    flux_state_entropy = _entropy_discrete(flux)

    # Iterate canonical dynamics from a subsample to estimate recurrent flux.
    S = S0[: min(len(S0), 512)].copy()
    flux_traj = []
    for t in range(int(horizon) + 1):
        B = _bits_from_states(S, partition)
        flux_traj.append(plaquette_flux_bits(lat, B))
        S = _step_states(lat.sys, S, lat.canonical_schedule)
    # Use second half as recurrent-ish diagnostic.
    rec = np.concatenate(flux_traj[max(1, len(flux_traj)//2):], axis=0)
    rec_entropy = _entropy_discrete(rec)
    # Period/fixed fraction diagnostic on flux along trajectories.
    F0 = flux_traj[-2] if len(flux_traj) >= 2 else flux_traj[0]
    F1 = flux_traj[-1]
    rec_fixed_fraction = float(np.all(F0 == F1, axis=1).mean()) if len(F0) else math.nan
    return dict(
        link_schedule_consistency=link_sched,
        flux_schedule_consistency=flux_sched,
        quotient_determinism_accuracy=float(det_acc),
        quotient_determinism_exact=bool(det_exact),
        link_bit_entropy_bits=float(H_B),
        link_bit_entropy_norm=float(H_B / lat.n_links) if lat.n_links else 0.0,
        quotient_transition_mi_bits=float(MI_B),
        quotient_transition_mi_norm=float(MI_B / H_B) if H_B > 1e-12 else 0.0,
        plaquette_flux_entropy_bits=float(H_F1),
        initial_flux_entropy_bits=float(flux_state_entropy),
        recurrent_flux_entropy_bits=float(rec_entropy),
        recurrent_flux_fixed_fraction=rec_fixed_fraction,
        mean_nontrivial_flux_fraction=nontriv,
    )


def measure_closed_lattice(
    lat: ClosedPlaquetteLattice,
    n_initial: int = 4096,
    horizon: int = 24,
    schedule_samples: int = 8,
    rng: np.random.Generator | None = None,
    initial_mode: str = "joint_random",
    partitions: list[np.ndarray] | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    if partitions is None:
        partitions = binary_partitions(lat.q)
    S0 = _initial_states(lat, n_initial, rng, initial_mode)
    schedules = _random_schedules(lat.n_links, schedule_samples, rng, include_canonical=lat.canonical_schedule)
    best = None
    for pi, part in enumerate(partitions):
        m = _transition_metrics(lat, S0, part, schedules, horizon, rng)
        # Consistency-only quotient score.  No direct reward for plaquette flux
        # entropy or excitation lifetime.  Link-bit entropy prevents the trivial
        # collapsed quotient, but flux dynamics remains diagnostic.
        score = (
            0.35 * m["quotient_determinism_accuracy"]
            + 0.25 * m["link_schedule_consistency"]
            + 0.20 * m["flux_schedule_consistency"]
            + 0.20 * min(1.0, m["link_bit_entropy_norm"])
        )
        mm = dict(m)
        mm.update(partition_index=int(pi), partition="".join(str(int(x)) for x in part.tolist()), quotient_score=float(score))
        if best is None or mm["quotient_score"] > best["quotient_score"]:
            best = mm
    assert best is not None
    row = dict(
        q=int(lat.q), rows=int(lat.rows), cols=int(lat.cols), n_links=int(lat.n_links), n_plaquettes=int(lat.n_plaquettes),
        rule_ensemble=str(lat.meta.get("rule_ensemble", "unknown")),
        mutation_rate=float(lat.meta.get("mutation_rate", math.nan)),
        closed_scc=bool(lat.meta.get("closed_scc", False)),
        n_scc=int(lat.meta.get("n_scc", -1)),
    )
    row.update(best)
    selected = bool(row["closed_scc"] and row["quotient_determinism_accuracy"] >= 0.99 and row["link_schedule_consistency"] >= 0.95 and row["flux_schedule_consistency"] >= 0.95 and row["link_bit_entropy_norm"] > 0.2)
    row.update(
        closed_c2_quotient=selected,
        recurrent_flux_nontrivial=bool(best["recurrent_flux_entropy_bits"] > 0.1),
        closed_medium_candidate=bool(selected and best["recurrent_flux_entropy_bits"] > 0.1),
    )
    return row


# --------------------------------------------------------------------------- #
# Sweep controls
# --------------------------------------------------------------------------- #
def run_closed_sweep(
    q: int = 4,
    sizes: Iterable[tuple[int, int]] = ((2, 2),),
    ensembles: Iterable[Ensemble] = ("c2xor", "copy", "random"),
    mutation_rates: Iterable[float] = (0.0,),
    instances: int = 20,
    n_initial: int = 4096,
    horizon: int = 24,
    schedule_samples: int = 8,
    base_seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []
    for rows_n, cols_n in sizes:
        for ens in ensembles:
            for mu in mutation_rates:
                for inst in range(int(instances)):
                    seed = (int(base_seed) * 1000003 + int(q) * 9176 + int(rows_n) * 313 + int(cols_n) * 571 + hash(str(ens)) % 1000 * 37 + int(round(float(mu)*10000)) * 11 + inst) % 2**32
                    rng = np.random.default_rng(seed)
                    lat = make_closed_lattice(q, rows_n, cols_n, rng, ensemble=ens, mutation_rate=float(mu))
                    row = measure_closed_lattice(lat, n_initial=n_initial, horizon=horizon, schedule_samples=schedule_samples, rng=rng)
                    row.update(seed=int(seed), instance=int(inst), mode="sweep")
                    rows.append(row)
                if verbose:
                    print(f"closed-sweep size={rows_n}x{cols_n} ens={ens} mu={mu} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Blind rule search on closed topology
# --------------------------------------------------------------------------- #
def _mutate_lattice(lat: ClosedPlaquetteLattice, rng: np.random.Generator, entry_rate: float, table_rate: float) -> ClosedPlaquetteLattice:
    rules = [r.copy() for r in lat.sys.rules]
    q = lat.q
    for v, rule in enumerate(rules):
        if rng.random() < float(table_rate):
            rules[v] = _random_rule(q, len(lat.sys.preds[v]), rng)
        else:
            mask = rng.random(rule.size) < float(entry_rate)
            if np.any(mask):
                jumps = rng.integers(1, q, size=int(mask.sum()), dtype=np.int64)
                rule[mask] = (rule[mask] + jumps) % q
                rules[v] = rule
    sys = C.RelationalSystem(lat.sys.k, lat.sys.q, list(lat.sys.preds), rules, meta=dict(lat.sys.meta))
    return ClosedPlaquetteLattice(sys=sys, q=lat.q, rows=lat.rows, cols=lat.cols, H=lat.H.copy(), V=lat.V.copy(), canonical_schedule=lat.canonical_schedule, meta=dict(lat.meta))


def run_blind_closed_selection(
    q: int = 4,
    size: tuple[int, int] = (2, 2),
    null_samples: int = 100,
    runs: int = 5,
    population: int = 24,
    generations: int = 100,
    n_initial: int = 4096,
    horizon: int = 24,
    schedule_samples: int = 8,
    entry_rate: float = 0.08,
    table_rate: float = 0.05,
    base_seed: int = 0,
    save_winners: str | None = None,
    save_winner_top_n: int = 25,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []
    winner_pool: list[tuple[float, ClosedPlaquetteLattice, dict]] = []
    R, Cc = size
    # Null random rules.
    for i in range(int(null_samples)):
        seed = (int(base_seed) * 99991 + int(q) * 1237 + i) % 2**32
        rng = np.random.default_rng(seed)
        lat = make_closed_lattice(q, R, Cc, rng, ensemble="random")
        row = measure_closed_lattice(lat, n_initial=n_initial, horizon=horizon, schedule_samples=schedule_samples, rng=rng)
        row.update(search_mode="null", run=-1, generation=0, candidate=i, seed=int(seed))
        rows.append(row)
    if verbose:
        print(f"closed-blind null q={q} size={R}x{Cc} done", flush=True)
    # Evolution.
    for run in range(int(runs)):
        rng = np.random.default_rng((int(base_seed) * 88891 + int(q) * 7919 + run * 104729) % 2**32)
        pop = [make_closed_lattice(q, R, Cc, rng, ensemble="random") for _ in range(int(population))]
        scored: list[tuple[float, ClosedPlaquetteLattice, dict]] = []
        for gen in range(int(generations) + 1):
            scored = []
            for ci, lat in enumerate(pop):
                row = measure_closed_lattice(lat, n_initial=n_initial, horizon=horizon, schedule_samples=schedule_samples, rng=rng)
                fit = float(row["quotient_score"])
                # Mild bonus for exact closed quotient; recurrent flux remains
                # post-hoc and is deliberately not rewarded.
                if row["closed_c2_quotient"]:
                    fit += 0.25
                row.update(search_mode="evolve", run=int(run), generation=int(gen), candidate=int(ci), blind_fitness=float(fit))
                rows.append(row)
                scored.append((fit, lat, row))
                winner_pool.append((fit, lat, row))
            scored.sort(key=lambda z: z[0], reverse=True)
            best = scored[0][2]
            if verbose:
                print(f"closed-blind run={run} gen={gen} best={scored[0][0]:.4f} c2={best['closed_c2_quotient']} recH={best['recurrent_flux_entropy_bits']:.3f} det={best['quotient_determinism_accuracy']:.3f}", flush=True)
            if gen == int(generations):
                break
            elites = [x[1] for x in scored[: max(1, int(population)//4)]]
            new_pop = [e for e in elites]
            while len(new_pop) < int(population):
                parent = elites[int(rng.integers(0, len(elites)))]
                child = _mutate_lattice(parent, rng, entry_rate=entry_rate, table_rate=table_rate)
                new_pop.append(child)
            pop = new_pop
    # Save top unique winners.
    if save_winners:
        os.makedirs(os.path.dirname(save_winners) or ".", exist_ok=True)
        winner_pool.sort(key=lambda z: z[0], reverse=True)
        packed = []
        seen = set()
        for fit, lat, row in winner_pool:
            key = tuple(tuple(int(x) for x in r.tolist()) for r in lat.sys.rules)
            if key in seen:
                continue
            seen.add(key)
            packed.append(dict(fitness=float(fit), row=dict(row), lattice=lat))
            if len(packed) >= int(save_winner_top_n):
                break
        with open(save_winners, "wb") as f:
            pickle.dump(packed, f)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Analysis / plotting
# --------------------------------------------------------------------------- #
def analyze_closed(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    final = df.copy()
    if "search_mode" in df.columns and (df["search_mode"] == "evolve").any():
        evo = df[df.search_mode == "evolve"]
        maxg = evo.groupby("run")["generation"].max().to_dict()
        final = evo[evo.apply(lambda r: int(r.generation) == int(maxg.get(r.run, -1)), axis=1)]
        null = df[df.search_mode == "null"]
    else:
        null = df.iloc[0:0]
    closed_frac = float(df["closed_c2_quotient"].mean()) if "closed_c2_quotient" in df else 0.0
    medium_frac = float(df["closed_medium_candidate"].mean()) if "closed_medium_candidate" in df else 0.0
    final_closed = float(final["closed_c2_quotient"].mean()) if len(final) and "closed_c2_quotient" in final else 0.0
    final_medium = float(final["closed_medium_candidate"].mean()) if len(final) and "closed_medium_candidate" in final else 0.0
    null_closed = float(null["closed_c2_quotient"].mean()) if len(null) else math.nan
    if final_medium > 0:
        verdict = "CLOSED SCC C2 MEDIUM SIGNAL: closed recurrent lattices support C2 quotients with nontrivial recurrent flux"
    elif final_closed > 0 or closed_frac > 0:
        verdict = "CLOSED SCC C2 QUOTIENT SIGNAL: binary quotient survives closure but recurrent flux is mostly vacuum/static"
    else:
        verdict = "NO CLOSED SCC C2 QUOTIENT SIGNAL in this regime"
    by = []
    if "rule_ensemble" in df.columns:
        for (ens, rows, cols), g in df.groupby(["rule_ensemble", "rows", "cols"]):
            by.append(dict(
                rule_ensemble=str(ens), rows=int(rows), cols=int(cols), n=int(len(g)),
                closed_c2_fraction=float(g.closed_c2_quotient.mean()),
                closed_medium_fraction=float(g.closed_medium_candidate.mean()),
                mean_quotient_score=float(g.quotient_score.mean()),
                mean_det_accuracy=float(g.quotient_determinism_accuracy.mean()),
                mean_link_schedule=float(g.link_schedule_consistency.mean()),
                mean_flux_schedule=float(g.flux_schedule_consistency.mean()),
                mean_recurrent_flux_entropy_bits=float(g.recurrent_flux_entropy_bits.mean()),
                mean_nontrivial_flux_fraction=float(g.mean_nontrivial_flux_fraction.mean()),
            ))
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        closed_c2_fraction=closed_frac,
        closed_medium_fraction=medium_frac,
        null_closed_c2_fraction=null_closed,
        final_closed_c2_fraction=final_closed,
        final_closed_medium_fraction=final_medium,
        all_time_best_score=float(df.get("blind_fitness", df["quotient_score"]).max()) if len(df) else math.nan,
        all_time_best_recurrent_flux_entropy=float(df.loc[df.get("blind_fitness", df["quotient_score"]).idxmax(), "recurrent_flux_entropy_bits"]) if len(df) else math.nan,
        by_ensemble=by,
    )


def plot_closed(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    if "search_mode" in df.columns and (df.search_mode == "evolve").any():
        evo = df[df.search_mode == "evolve"]
        g = evo.groupby("generation")
        ax[0].plot(g["blind_fitness"].max(), marker="o", label="global best")
        ax[0].plot(g["blind_fitness"].mean(), marker="o", label="mean")
        ax[0].set_title("closed SCC fitness"); ax[0].set_xlabel("generation"); ax[0].set_ylabel("fitness"); ax[0].legend()
        ax[1].plot(g["closed_c2_quotient"].mean(), marker="o", label="C2 quotient")
        ax[1].plot(g["closed_medium_candidate"].mean(), marker="o", label="recurrent medium")
        ax[1].set_title("closed quotient over time"); ax[1].set_xlabel("generation"); ax[1].set_ylabel("fraction"); ax[1].legend()
        ax[2].scatter(evo["link_schedule_consistency"], evo["recurrent_flux_entropy_bits"], c=evo["closed_c2_quotient"].astype(int), alpha=0.7)
        ax[2].set_title("schedule consistency vs flux entropy"); ax[2].set_xlabel("link schedule consistency"); ax[2].set_ylabel("recurrent flux entropy")
    else:
        labels = [] ; c2 = [] ; med = [] ; H = []
        for ens, g in df.groupby("rule_ensemble"):
            labels.append(str(ens)); c2.append(float(g.closed_c2_quotient.mean())); med.append(float(g.closed_medium_candidate.mean())); H.append(float(g.recurrent_flux_entropy_bits.mean()))
        x = np.arange(len(labels))
        ax[0].bar(x, c2); ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=30); ax[0].set_title("closed C2 quotient fraction")
        ax[1].bar(x, med); ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=30); ax[1].set_title("closed recurrent medium fraction")
        ax[2].bar(x, H); ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=30); ax[2].set_title("recurrent flux entropy")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Closed SCC plaquette lattice audit/search.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--mode", choices=["sweep", "blind"], default="sweep")
    p.add_argument("--sizes", default="2x2")
    p.add_argument("--ensembles", default="c2xor,copy,random")
    p.add_argument("--mutation-rates", default="0")
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--n-initial", type=int, default=4096)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--schedule-samples", type=int, default=8)
    p.add_argument("--null-samples", type=int, default=100)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--entry-rate", type=float, default=0.08)
    p.add_argument("--table-rate", type=float, default=0.05)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--save-winners", default=None)
    p.add_argument("--save-winner-top-n", type=int, default=25)
    p.add_argument("--out", default="example_results/closed_plaquette.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    sizes = [_parse_size(x) for x in _parse_list(args.sizes, str)]
    if args.mode == "sweep":
        df = run_closed_sweep(
            q=int(args.q),
            sizes=sizes,
            ensembles=_parse_list(args.ensembles, str),
            mutation_rates=_parse_list(args.mutation_rates, float),
            instances=int(args.instances),
            n_initial=int(args.n_initial),
            horizon=int(args.horizon),
            schedule_samples=int(args.schedule_samples),
            base_seed=int(args.base_seed),
            verbose=not args.quiet,
        )
    else:
        if len(sizes) != 1:
            raise SystemExit("--mode blind currently accepts exactly one --sizes entry")
        df = run_blind_closed_selection(
            q=int(args.q),
            size=sizes[0],
            null_samples=int(args.null_samples),
            runs=int(args.runs),
            population=int(args.population),
            generations=int(args.generations),
            n_initial=int(args.n_initial),
            horizon=int(args.horizon),
            schedule_samples=int(args.schedule_samples),
            entry_rate=float(args.entry_rate),
            table_rate=float(args.table_rate),
            base_seed=int(args.base_seed),
            save_winners=args.save_winners,
            save_winner_top_n=int(args.save_winner_top_n),
            verbose=not args.quiet,
        )
    summary = analyze_closed(df)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_closed(df, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
