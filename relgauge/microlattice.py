"""
microlattice.py -- microscopic coupled Z2/shared-corner lattice and Wilson loops.

This module is the microscopic counterpart to the earlier effective
``wilsonlattice.py`` experiment.  The earlier experiment sampled plaquette
fluxes independently from the empirical blind shared-square distribution; an
area law follows automatically from independent plaquette variables.  Here the
plaquettes are not sampled independently.  We build one feed-forward square
lattice in which neighbouring plaquettes literally share transported edge
systems.

The orientation is acyclic: edges point east and south.  Each directed edge is
implemented as a one-diamond transporter from a source corner output to two
sink panels at the target corner.  Adjacent plaquettes share the same edge
transporters.  At each plaquette we infer four transported label maps and the
finite path holonomy

    Delta = (phi_right o phi_top)^(-1) o (phi_bottom o phi_left).

For C2/Z2 plaquettes, identity is encoded as +1 and the swap as -1.  Wilson
loops are then computed as products of microscopic plaquette fluxes inside a
rectangular loop.  Because fluxes arise from shared edges in one coupled
lattice, an area law is no longer automatic.

Recommended quick run
---------------------
python -m relgauge.microlattice 4 ^
  --sizes 4x4,6x6 ^
  --instances 100 ^
  --ensembles c2link,copy,permutive,random ^
  --p-flips 0.1,0.3,0.5 ^
  --max-loop-side 4 ^
  --out example_results/microlattice_q4_w1.csv ^
  --plot example_results/fig_microlattice_q4_w1.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

from . import core as C
from .sharedsquareholonomy import (
    _random_rule,
    _mutate_rule,
    _set_copy,
    _set_constant,
    _perm_transform,
    _block_transform,
    _canalizing_transform,
    _encode_words_from_state,
    exact_equalizer_labels,
    _best_permutation_map,
    compose,
    inverse,
    identity,
    cycle_type_name,
    generate_group,
    classify_group,
)

Ensemble = Literal["c2link", "copy", "permutive", "block", "canalizing", "random", "constant"]
InitialMode = Literal["source_all", "joint_random"]


@dataclass
class EdgeData:
    name: str
    src: tuple[int, int]
    dst: tuple[int, int]
    source: tuple[int, ...]
    branch_left: tuple[int, ...]
    branch_right: tuple[int, ...]
    panel_left: tuple[int, ...]
    panel_right: tuple[int, ...]
    flip: int = 0


@dataclass
class MicroLattice:
    joint: C.RelationalSystem
    q: int
    w: int
    nx: int
    ny: int
    node_out: dict[tuple[int, int], tuple[int, ...]]
    edges: dict[tuple[tuple[int, int], tuple[int, int]], EdgeData]
    plaquettes: list[tuple[int, int]]
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


def _connect(srcs: tuple[int, ...], tgts: tuple[int, ...], pred_sets: list[set[int]]) -> None:
    for s, t in zip(srcs, tgts):
        pred_sets[int(t)].add(int(s))


def _rule_from_source_func(q: int, indeg: int, pos: int, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    X = C.all_states(int(indeg), int(q)) if indeg else np.zeros((1, 0), dtype=np.int64)
    y = func(X[:, int(pos)] if indeg else np.zeros(1, dtype=np.int64))
    return np.asarray(y, dtype=np.int64) % int(q)


def _set_copy_func(sys: C.RelationalSystem, target: int, source: int, func: Callable[[np.ndarray], np.ndarray]) -> None:
    preds = tuple(int(x) for x in sys.preds[int(target)])
    pos = preds.index(int(source))
    sys.rules[int(target)] = _rule_from_source_func(sys.q, len(preds), pos, func)


def _binary_func(flip: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    # Coarse binary label: parity, optionally flipped.  For q>2 this maps the
    # microscopic alphabet to a stable two-symbol transported factor.
    return lambda x: ((np.asarray(x, dtype=np.int64) % 2) ^ int(flip)).astype(np.int64)


def _edge_key(src: tuple[int, int], dst: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (tuple(src), tuple(dst))  # type: ignore[return-value]


def make_microscopic_lattice(
    q: int = 4,
    nx: int = 4,
    ny: int = 4,
    w: int = 1,
    rng: np.random.Generator | None = None,
    ensemble: Ensemble = "c2link",
    p_flip: float = 0.5,
    mutation_rate: float = 0.0,
    n_blocks: int | None = None,
) -> MicroLattice:
    """Build one coupled feed-forward square lattice.

    ``nx`` and ``ny`` are the numbers of plaquettes in the horizontal and
    vertical directions.  There are (ny+1)*(nx+1) corner outputs.  Directed
    edges point east and south.  Each edge is a one-diamond transporter.  Each
    non-source corner output copies a preferred incoming panel, so the corner is
    physically shared by its incoming and outgoing edges.
    """
    q = int(q); nx = int(nx); ny = int(ny); w = int(w)
    if rng is None:
        rng = np.random.default_rng(0)
    if nx < 1 or ny < 1 or w < 1:
        raise ValueError("nx, ny, w must be positive")

    pred_sets: list[set[int]] = []
    node_out: dict[tuple[int, int], tuple[int, ...]] = {}
    for r in range(ny + 1):
        for c in range(nx + 1):
            node_out[(r, c)] = _add_vertices(w, pred_sets)

    edges: dict[tuple[tuple[int, int], tuple[int, int]], EdgeData] = {}
    # Horizontal and vertical directed edges.
    for r in range(ny + 1):
        for c in range(nx):
            src, dst = (r, c), (r, c + 1)
            bl = _add_vertices(w, pred_sets); br = _add_vertices(w, pred_sets)
            pl = _add_vertices(w, pred_sets); pr = _add_vertices(w, pred_sets)
            _connect(node_out[src], bl, pred_sets); _connect(node_out[src], br, pred_sets)
            _connect(bl, pl, pred_sets); _connect(br, pr, pred_sets)
            flip = int(rng.random() < float(p_flip))
            edges[_edge_key(src, dst)] = EdgeData(f"H{r}_{c}", src, dst, node_out[src], bl, br, pl, pr, flip=flip)
    for r in range(ny):
        for c in range(nx + 1):
            src, dst = (r, c), (r + 1, c)
            bl = _add_vertices(w, pred_sets); br = _add_vertices(w, pred_sets)
            pl = _add_vertices(w, pred_sets); pr = _add_vertices(w, pred_sets)
            _connect(node_out[src], bl, pred_sets); _connect(node_out[src], br, pred_sets)
            _connect(bl, pl, pred_sets); _connect(br, pr, pred_sets)
            flip = int(rng.random() < float(p_flip))
            edges[_edge_key(src, dst)] = EdgeData(f"V{r}_{c}", src, dst, node_out[src], bl, br, pl, pr, flip=flip)

    # Corner outputs depend on a preferred incoming panel, except the root
    # (0,0), whose source vertices have no predecessors and act as exogenous
    # one-tick source variables.  Preference order: west, then north.
    for r in range(ny + 1):
        for c in range(nx + 1):
            if (r, c) == (0, 0):
                continue
            incoming: EdgeData | None = None
            if c > 0:
                incoming = edges[_edge_key((r, c - 1), (r, c))]
            elif r > 0:
                incoming = edges[_edge_key((r - 1, c), (r, c))]
            if incoming is not None:
                _connect(incoming.panel_left, node_out[(r, c)], pred_sets)

    preds = [tuple(sorted(s)) for s in pred_sets]
    rules = [_random_rule(q, len(preds[v]), rng) for v in range(len(preds))]
    sys = C.RelationalSystem(len(preds), q, preds, rules, meta=dict(ensemble="micro_lattice"))

    # Choose transforms for edge branches and corner outputs.
    def set_edge(ed: EdgeData, left_func: Callable[[np.ndarray], np.ndarray] | None, right_func: Callable[[np.ndarray], np.ndarray] | None) -> None:
        for s, t in zip(ed.source, ed.branch_left):
            if left_func is None:
                _set_copy(sys, int(t), int(s), None)
            else:
                _set_copy_func(sys, int(t), int(s), left_func)
        for s, t in zip(ed.source, ed.branch_right):
            if right_func is None:
                _set_copy(sys, int(t), int(s), None)
            else:
                _set_copy_func(sys, int(t), int(s), right_func)
        for s, t in zip(ed.branch_left, ed.panel_left):
            _set_copy(sys, int(t), int(s), None)
        for s, t in zip(ed.branch_right, ed.panel_right):
            _set_copy(sys, int(t), int(s), None)

    if ensemble == "copy":
        for ed in edges.values():
            set_edge(ed, None, None)
    elif ensemble == "c2link":
        for ed in edges.values():
            f = _binary_func(ed.flip)
            set_edge(ed, f, f)
    elif ensemble == "permutive":
        for ed in edges.values():
            pL = rng.permutation(q).astype(np.int64)
            pR = rng.permutation(q).astype(np.int64)
            set_edge(ed, _perm_transform(pL), _perm_transform(pR))
    elif ensemble == "block":
        blk = _block_transform(q, n_blocks)
        for ed in edges.values():
            set_edge(ed, blk, blk)
    elif ensemble == "canalizing":
        can = _canalizing_transform(q)
        for ed in edges.values():
            set_edge(ed, can, can)
    elif ensemble == "constant":
        for ed in edges.values():
            for v in (*ed.branch_left, *ed.branch_right, *ed.panel_left, *ed.panel_right):
                _set_constant(sys, int(v), 0)
    elif ensemble == "random":
        pass
    else:
        raise ValueError(f"unknown ensemble {ensemble!r}")

    # Corner output rules.  For c2link and coarse ensembles this propagates the
    # selected label; for permutive/copy it copies the preferred incoming panel.
    for r in range(ny + 1):
        for c in range(nx + 1):
            if (r, c) == (0, 0):
                continue
            incoming = None
            if c > 0:
                incoming = edges[_edge_key((r, c - 1), (r, c))]
            elif r > 0:
                incoming = edges[_edge_key((r - 1, c), (r, c))]
            if incoming is None:
                continue
            for s, t in zip(incoming.panel_left, node_out[(r, c)]):
                _set_copy(sys, int(t), int(s), None)

    if mutation_rate > 0:
        for v in range(sys.k):
            if len(sys.preds[v]) > 0:
                sys.rules[v] = _mutate_rule(sys.rules[v], q, float(mutation_rate), rng)

    # Schedule by topological layers: node outputs already have predecessors for
    # non-root nodes, so we use a conservative hand-ordered pass from NW to SE.
    sched: list[int] = []
    # root first
    sched.extend(node_out[(0, 0)])
    # Sweep diagonals. At each node output, then its outgoing edge branches and panels.
    for sdiag in range(nx + ny + 1):
        for r in range(max(0, sdiag - nx), min(ny, sdiag) + 1):
            c = sdiag - r
            if c < 0 or c > nx:
                continue
            # update corner output after incoming panels from previous diagonal
            if (r, c) != (0, 0):
                sched.extend(node_out[(r, c)])
            for dst in ((r, c + 1), (r + 1, c)):
                if dst in node_out and _edge_key((r, c), dst) in edges:
                    ed = edges[_edge_key((r, c), dst)]
                    sched.extend(ed.branch_left); sched.extend(ed.branch_right)
                    sched.extend(ed.panel_left); sched.extend(ed.panel_right)
    seen = set(sched)
    sched.extend(v for v in range(sys.k) if v not in seen)

    plaquettes = [(r, c) for r in range(ny) for c in range(nx)]
    return MicroLattice(
        joint=sys, q=q, w=w, nx=nx, ny=ny, node_out=node_out, edges=edges,
        plaquettes=plaquettes, schedule=tuple(int(v) for v in sched),
        meta=dict(ensemble=ensemble, p_flip=float(p_flip), mutation_rate=float(mutation_rate)),
    )


def _initial_states(lat: MicroLattice, mode: InitialMode, n_random_initial: int, rng: np.random.Generator) -> np.ndarray:
    q, w, k = int(lat.q), int(lat.w), int(lat.joint.k)
    source_words = np.arange(q ** w, dtype=np.int64)
    if mode == "source_all":
        N = len(source_words)
        S = np.zeros((N, k), dtype=np.int64)
        reps = source_words
    elif mode == "joint_random":
        N = max(int(n_random_initial), len(source_words))
        S = rng.integers(0, q, size=(N, k), dtype=np.int64)
        reps = np.resize(source_words, N)
        rng.shuffle(reps)
    else:
        raise ValueError(f"unknown initial mode {mode!r}")
    root = lat.node_out[(0, 0)]
    for j, v in enumerate(root):
        S[:, int(v)] = (reps // (q ** j)) % q
    return S


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


def _path_delta(ab: tuple[int, ...], bd: tuple[int, ...], ac: tuple[int, ...], cd: tuple[int, ...]) -> dict:
    top = compose(bd, ab)
    bottom = compose(cd, ac)
    delta = compose(inverse(top), bottom)
    return dict(top=top, bottom=bottom, delta=delta)


def _edge_label_arrays(lat: MicroLattice, S1: np.ndarray) -> dict:
    labels = {}
    infos = {}
    for key, ed in lat.edges.items():
        left = _encode_words_from_state(S1, ed.panel_left, lat.q)
        right = _encode_words_from_state(S1, ed.panel_right, lat.q)
        eq = exact_equalizer_labels(left, right, lat.q, lat.w)
        labels[key] = np.asarray(eq["labels"], dtype=np.int64)
        infos[key] = eq
    return labels, infos


def _plaquette_measure(lat: MicroLattice, labels: dict, r: int, c: int, min_transport: float, min_accuracy: float) -> dict:
    # Edge labels around plaquette A=(r,c), B=(r,c+1), C=(r+1,c), D=(r+1,c+1)
    kAB = _edge_key((r, c), (r, c + 1))
    kAC = _edge_key((r, c), (r + 1, c))
    kBD = _edge_key((r, c + 1), (r + 1, c + 1))
    kCD = _edge_key((r + 1, c), (r + 1, c + 1))
    SAB = labels[kAB]
    SAC = labels[kAC]
    SBD = labels[kBD]
    SCD = labels[kCD]
    n_classes = int(len(np.unique(SAB)))
    phi_AB = tuple(range(n_classes))
    mAC = _best_permutation_map(SAB, SAC)
    mBD = _best_permutation_map(SAB, SBD)
    mCD = _best_permutation_map(SAC, SCD)
    maps = [phi_AB, mAC["permutation"], mBD["permutation"], mCD["permutation"]]
    min_trans = float(min(mAC["mi_over_Hy"], mBD["mi_over_Hy"], mCD["mi_over_Hy"]))
    min_acc = float(min(mAC["accuracy"], mBD["accuracy"], mCD["accuracy"]))
    valid = bool(n_classes > 1 and all(p is not None and len(p) == n_classes for p in maps) and min_trans >= min_transport and min_acc >= min_accuracy)
    out = dict(r=int(r), c=int(c), n_classes=n_classes, valid=valid, min_transport=min_trans, min_accuracy=min_acc)
    if not valid:
        out.update(delta_type="invalid", flux=np.nan, generated_group_name="invalid")
        return out
    ab = tuple(int(x) for x in phi_AB)
    bd = tuple(int(x) for x in mBD["permutation"])
    ac = tuple(int(x) for x in mAC["permutation"])
    cd = tuple(int(x) for x in mCD["permutation"])
    pd = _path_delta(ab, bd, ac, cd)
    delta = pd["delta"]
    group = generate_group([ab, bd, ac, cd], n=n_classes)
    gsum = classify_group(group, n=n_classes)
    dtype = cycle_type_name(delta)
    flux = np.nan
    if n_classes == 2 and dtype in ("identity", "cycle_2"):
        flux = 1.0 if dtype == "identity" else -1.0
    out.update(
        delta_type=dtype,
        nontrivial=bool(delta != identity(n_classes)),
        flux=flux,
        generated_group_name=str(gsum.get("group_name")),
        generated_group_order=int(gsum.get("group_order", 0)),
    )
    return out


def measure_microscopic_lattice(
    lat: MicroLattice,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    min_transport: float = 0.80,
    min_accuracy: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    S0 = _initial_states(lat, initial_mode, n_random_initial, rng)
    S1 = _step_states(lat.joint, S0, lat.schedule)
    labels, edge_infos = _edge_label_arrays(lat, S1)
    plaquettes = [_plaquette_measure(lat, labels, r, c, min_transport, min_accuracy) for r, c in lat.plaquettes]
    valid = [p for p in plaquettes if p["valid"]]
    z2 = [p for p in valid if not math.isnan(float(p["flux"]))]
    flux_grid = np.full((lat.ny, lat.nx), np.nan, dtype=float)
    for p in z2:
        flux_grid[int(p["r"]), int(p["c"])] = float(p["flux"])
    return dict(
        plaquettes=plaquettes,
        flux_grid=flux_grid,
        valid_fraction=float(len(valid) / max(1, len(plaquettes))),
        z2_fraction=float(len(z2) / max(1, len(plaquettes))),
        nontrivial_fraction=float(np.mean([p["nontrivial"] for p in valid])) if valid else 0.0,
        group_counts={str(k): int(v) for k, v in Counter(p["generated_group_name"] for p in valid).items()},
        delta_counts={str(k): int(v) for k, v in Counter(p["delta_type"] for p in valid).items()},
    )


def _rect_wilson_values(flux_grid: np.ndarray, a: int, b: int) -> list[float]:
    ny, nx = flux_grid.shape
    vals = []
    for r in range(0, ny - int(b) + 1):
        for c in range(0, nx - int(a) + 1):
            sub = flux_grid[r:r+int(b), c:c+int(a)]
            if np.all(np.isfinite(sub)):
                vals.append(float(np.prod(sub)))
    return vals


def run_microlattice_sweep(
    q: int = 4,
    sizes: Iterable[tuple[int, int]] = ((4, 4),),
    w: int = 1,
    ensembles: Iterable[Ensemble] = ("c2link", "copy", "permutive", "random"),
    p_flips: Iterable[float] = (0.1, 0.3, 0.5),
    mutation_rates: Iterable[float] = (0.0,),
    instances: int = 100,
    max_loop_side: int = 4,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    base_seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for nx, ny in sizes:
        for ens in ensembles:
            pflist = list(p_flips) if ens == "c2link" else [0.0]
            for pflip in pflist:
                for mu in mutation_rates:
                    loop_acc: dict[tuple[int,int], list[float]] = defaultdict(list)
                    valid_fracs = []
                    z2_fracs = []
                    nontriv_fracs = []
                    delta_counts = Counter()
                    group_counts = Counter()
                    for inst in range(int(instances)):
                        seed = (int(base_seed) * 1000003 + int(q) * 10007 + int(nx) * 1009 + int(ny) * 917 + hash(str(ens)) % 1000 * 37 + int(round(float(pflip)*1000))*13 + int(round(float(mu)*10000))*11 + inst) % 2**32
                        rng = np.random.default_rng(seed)
                        lat = make_microscopic_lattice(q=q, nx=nx, ny=ny, w=w, rng=rng, ensemble=ens, p_flip=pflip, mutation_rate=mu)
                        meas = measure_microscopic_lattice(lat, initial_mode=initial_mode, n_random_initial=n_random_initial, rng=rng)
                        valid_fracs.append(meas["valid_fraction"]); z2_fracs.append(meas["z2_fraction"]); nontriv_fracs.append(meas["nontrivial_fraction"])
                        delta_counts.update(meas["delta_counts"]); group_counts.update(meas["group_counts"])
                        F = meas["flux_grid"]
                        for a in range(1, min(max_loop_side, nx) + 1):
                            for b in range(1, min(max_loop_side, ny) + 1):
                                loop_acc[(a,b)].extend(_rect_wilson_values(F, a, b))
                    for (a,b), vals in loop_acc.items():
                        vals = list(vals)
                        meanW = float(np.mean(vals)) if vals else math.nan
                        absW = float(abs(meanW)) if vals else math.nan
                        rows.append(dict(
                            q=int(q), w=int(w), nx=int(nx), ny=int(ny), ensemble=str(ens), p_flip=float(pflip), mutation_rate=float(mu),
                            instances=int(instances), loop_a=int(a), loop_b=int(b), area=int(a*b), perimeter=int(2*(a+b)),
                            n_loop_samples=int(len(vals)), mean_W=meanW, abs_mean_W=absW,
                            valid_plaquette_fraction=float(np.mean(valid_fracs)), z2_plaquette_fraction=float(np.mean(z2_fracs)),
                            nontrivial_plaquette_fraction=float(np.mean(nontriv_fracs)),
                            delta_type_counts=json.dumps({str(k): int(v) for k,v in delta_counts.items()}),
                            group_name_counts=json.dumps({str(k): int(v) for k,v in group_counts.items()}),
                        ))
                    if verbose:
                        print(f"microlattice ens={ens} p={pflip} mu={mu} size={nx}x{ny} done", flush=True)
    return pd.DataFrame(rows)


def analyze_microlattice(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    by = []
    for (ens, pflip, mu), g in df.groupby(["ensemble", "p_flip", "mutation_rate"]):
        # Fit log |W| against area and perimeter for rows with usable means.
        gg = g[np.isfinite(g["abs_mean_W"]) & (g["abs_mean_W"] > 1e-12)]
        area_alpha = perim_beta = math.nan
        area_rmse = perim_rmse = math.nan
        if len(gg) >= 2:
            y = -np.log(gg["abs_mean_W"].to_numpy(dtype=float))
            A = gg["area"].to_numpy(dtype=float)
            P = gg["perimeter"].to_numpy(dtype=float)
            area_alpha = float(np.dot(A, y) / max(np.dot(A, A), 1e-12))
            perim_beta = float(np.dot(P, y) / max(np.dot(P, P), 1e-12))
            area_rmse = float(np.sqrt(np.mean((y - area_alpha*A)**2)))
            perim_rmse = float(np.sqrt(np.mean((y - perim_beta*P)**2)))
        by.append(dict(
            ensemble=str(ens), p_flip=float(pflip), mutation_rate=float(mu),
            mean_valid_plaquette_fraction=float(g["valid_plaquette_fraction"].mean()),
            mean_z2_plaquette_fraction=float(g["z2_plaquette_fraction"].mean()),
            mean_nontrivial_plaquette_fraction=float(g["nontrivial_plaquette_fraction"].mean()),
            area_alpha=area_alpha, perimeter_beta=perim_beta, area_rmse=area_rmse, perimeter_rmse=perim_rmse,
        ))
    # Verdict emphasizes whether this is truly coupled and whether the loops are perimeter-like.
    c2 = [r for r in by if r["ensemble"] == "c2link"]
    if c2:
        best = max(c2, key=lambda r: r["mean_z2_plaquette_fraction"])
        if best["mean_z2_plaquette_fraction"] > 0.8 and (best["perimeter_rmse"] <= best["area_rmse"] or math.isnan(best["area_rmse"])):
            verdict = "COUPLED Z2 LINK-LATTICE: microscopic shared edges produce Wilson loops not reducible to independent plaquette sampling"
        elif best["mean_z2_plaquette_fraction"] > 0.8:
            verdict = "COUPLED Z2 PLAQUETTES: valid microscopic fluxes measured; compare area/perimeter fits"
        else:
            verdict = "PARTIAL MICROSCOPIC LATTICE SIGNAL"
    else:
        verdict = "MICROSCOPIC LATTICE SWEEP COMPLETE"
    return dict(verdict=verdict, n_rows=int(len(df)), by_ensemble=by)


def plot_microlattice(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    # Plot Wilson values vs area for each ensemble/p.
    for (ens, pflip), g in df.groupby(["ensemble", "p_flip"]):
        gg = g.groupby("area")["abs_mean_W"].mean().dropna()
        if len(gg):
            ax[0].plot(gg.index, gg.values, "o-", label=f"{ens} p={pflip:g}")
    ax[0].set_title("Wilson loop |<W>| vs area")
    ax[0].set_xlabel("area"); ax[0].set_ylabel("|<W>|")
    ax[0].legend(fontsize=7)
    for (ens, pflip), g in df.groupby(["ensemble", "p_flip"]):
        gg = g.groupby("perimeter")["abs_mean_W"].mean().dropna()
        if len(gg):
            ax[1].plot(gg.index, gg.values, "o-", label=f"{ens} p={pflip:g}")
    ax[1].set_title("Wilson loop |<W>| vs perimeter")
    ax[1].set_xlabel("perimeter"); ax[1].set_ylabel("|<W>|")
    # Valid plaquette fractions.
    labels = [] ; vals = []
    for (ens, pflip), g in df.groupby(["ensemble", "p_flip"]):
        labels.append(f"{ens}\np={pflip:g}")
        vals.append(float(g["z2_plaquette_fraction"].mean()))
    ax[2].bar(labels, vals)
    ax[2].set_title("valid C2 plaquette fraction")
    ax[2].tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _parse_sizes(s: str) -> list[tuple[int,int]]:
    out = []
    for part in str(s).split(','):
        part = part.strip().lower()
        if not part:
            continue
        if 'x' in part:
            a,b = part.split('x',1)
            out.append((int(a), int(b)))
        else:
            n = int(part); out.append((n,n))
    return out


def _parse_list(s: str, typ=str):
    if not s:
        return []
    return [typ(x) for x in str(s).split(',') if str(x).strip()]


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Microscopic coupled Z2 lattice / Wilson-loop test.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--sizes", default="4x4")
    p.add_argument("--w", type=int, default=1)
    p.add_argument("--instances", type=int, default=100)
    p.add_argument("--ensembles", default="c2link,copy,permutive,random")
    p.add_argument("--p-flips", default="0.1,0.3,0.5")
    p.add_argument("--mutation-rates", default="0")
    p.add_argument("--max-loop-side", type=int, default=4)
    p.add_argument("--initial-mode", choices=["source_all", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", default="example_results/microlattice.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    df = run_microlattice_sweep(
        q=int(args.q), sizes=_parse_sizes(args.sizes), w=int(args.w),
        ensembles=_parse_list(args.ensembles, str), p_flips=_parse_list(args.p_flips, float),
        mutation_rates=_parse_list(args.mutation_rates, float), instances=int(args.instances),
        max_loop_side=int(args.max_loop_side), initial_mode=args.initial_mode,
        n_random_initial=int(args.n_random_initial), base_seed=int(args.base_seed), verbose=not bool(args.quiet),
    )
    summary = analyze_microlattice(df)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or '.', exist_ok=True)
        plot_microlattice(df, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
