"""
recursivefluxaudit.py -- recursive consistency audit on selected Z2 flux lattices.

This diagnostic applies the same finite/equalizer logic one level up.  The input
is not a hand-written matter field.  It is a saved microscopic lattice winner
from ``blindmicrolattice.py`` or ``latticegrowth.py``.  For each winner we first
read the selected C2/Z2 plaquette flux grid, then treat that flux grid as a new
finite relational object and ask whether it contains nontrivial local/common
factors or defect-like parity structure.

Default mode is algebraic: it does not iterate the microscopic rules.  It
searches the flux configuration itself for recursive local consistency
constraints:

  * adjacency equalizers between neighbouring plaquette fluxes;
  * parity fields on 1-plaquette, 2-plaquette, and 2x2-block patches;
  * equalizers between neighbouring patch-parity cells;
  * vertex-star parity diagnostics, interpreted only as post-hoc defect
    candidates, not inserted charges.

The optional trajectory mode is intentionally conservative.  It computes a
state-dependent edge-label-parity flux time series using the selected edge-label
maps frozen from a reference ensemble.  This is a diagnostic for dynamical
flux-level persistence, not the primary algebraic closure test.

Example
-------
python -m relgauge.recursivefluxaudit example_results/blind_microlattice_winners_q4_3x3.pkl ^
  --mode algebraic ^
  --top-n 50 ^
  --patch-sizes 1,2,4 ^
  --out example_results/recursive_flux_audit_q4_3x3.csv ^
  --plot example_results/fig_recursive_flux_audit_q4_3x3.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .microlattice import (
    MicroLattice,
    measure_microscopic_lattice,
    _edge_key,
    _initial_states,
    _step_states,
)
from .microlatticeaudit import flux_map_string, domain_stats, vertex_star_defect_proxy, all_wilson_rows, flux_correlation_rows
from .sharedsquareholonomy import exact_equalizer_labels, _encode_words_from_state

Mode = Literal["algebraic", "trajectory"]


# --------------------------------------------------------------------------- #
# loading and flux helpers
# --------------------------------------------------------------------------- #

def _load_winners(path: str) -> list[dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "winners" in obj:
        winners = obj["winners"]
    elif isinstance(obj, list):
        winners = obj
    else:
        raise ValueError("winner file must be a list or a dict with key 'winners'")
    if not isinstance(winners, list):
        raise ValueError("winners must be a list")
    return winners


def _winner_lattice(w: dict) -> MicroLattice:
    if isinstance(w, dict) and "lattice" in w:
        return w["lattice"]
    if isinstance(w, MicroLattice):
        return w
    raise ValueError("winner entry lacks a MicroLattice under key 'lattice'")


def _metric(w: dict, key: str, default=math.nan):
    if isinstance(w, dict):
        return w.get("metrics", {}).get(key, default)
    return default


def _flux_bits(F: np.ndarray) -> np.ndarray:
    """Map +1 -> 0, -1 -> 1, invalid -> -1."""
    F = np.asarray(F, dtype=float)
    B = np.full(F.shape, -1, dtype=np.int64)
    B[np.isfinite(F) & (F > 0)] = 0
    B[np.isfinite(F) & (F < 0)] = 1
    return B


def _bits_to_pm(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=np.int64)
    F = np.full(B.shape, np.nan, dtype=float)
    F[B == 0] = 1.0
    F[B == 1] = -1.0
    return F


def _encode_binary_words(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode rows of a binary matrix as words; rows with invalid entries are dropped.

    Returns (words, valid_mask).
    """
    A = np.asarray(arr, dtype=np.int64)
    if A.ndim == 1:
        A = A[:, None]
    valid = np.all(A >= 0, axis=1)
    if not np.any(valid):
        return np.array([], dtype=np.int64), valid
    V = A[valid]
    powers = (2 ** np.arange(V.shape[1], dtype=np.int64))[None, :]
    words = (V * powers).sum(axis=1).astype(np.int64)
    return words, valid


# --------------------------------------------------------------------------- #
# equalizer / patch diagnostics
# --------------------------------------------------------------------------- #

def _relation_type(pairs: np.ndarray, shared_classes: int) -> str:
    """Human-readable relation class for binary/binary relations."""
    if pairs.size == 0:
        return "empty"
    ps = {tuple(map(int, p)) for p in np.asarray(pairs, dtype=np.int64).reshape(-1, 2)}
    if shared_classes <= 1:
        return "connected/full-or-mixed"
    if ps <= {(0, 0), (1, 1)} and ps:
        return "same-label"
    if ps <= {(0, 1), (1, 0)} and ps:
        return "opposite-label"
    lefts = {p[0] for p in ps}; rights = {p[1] for p in ps}
    if len(lefts) == 1 or len(rights) == 1:
        return "one-sided/formal"
    return "disconnected/nontrivial"


def _equalizer_for_words(left: np.ndarray, right: np.ndarray, word_bits: int, label: str) -> dict:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    n = min(len(left), len(right))
    if n == 0:
        return dict(
            field=str(label), n_pairs=0, shared_classes=0, residual_bits=math.nan,
            residual_qcoords=math.nan, edge_density=math.nan, relation_type="empty",
        )
    left = left[:n]; right = right[:n]
    eq = exact_equalizer_labels(left, right, q=2, w=int(word_bits))
    pairs = np.asarray(eq.get("pairs", np.empty((0, 2), dtype=np.int64)), dtype=np.int64)
    shared = int(eq.get("shared_classes", 0))
    return dict(
        field=str(label), n_pairs=int(len(left)), word_bits=int(word_bits),
        shared_classes=shared,
        residual_bits=float(math.log2(shared)) if shared > 0 else math.nan,
        residual_qcoords=float(eq.get("residual_qcoords", math.nan)),
        edge_density=float(eq.get("edge_density", math.nan)),
        relation_type=_relation_type(pairs, shared) if word_bits == 1 else ("nontrivial" if shared > 1 else "connected/formal"),
        pair_values=";".join(f"{int(a)}:{int(b)}" for a, b in pairs.tolist()) if len(pairs) <= 64 else "many",
    )


def _adjacent_word_pairs(B: np.ndarray, block_h: int, block_w: int, direction: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Neighbouring patch-word pairs in a binary plaquette grid.

    ``block_h`` and ``block_w`` define the local patch.  Adjacent patches are
    shifted by one plaquette in the chosen direction.
    """
    B = np.asarray(B, dtype=np.int64)
    ny, nx = B.shape
    bh, bw = int(block_h), int(block_w)
    if bh <= 0 or bw <= 0 or ny < bh or nx < bw:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), bh * bw
    left_blocks = []
    right_blocks = []
    if direction == "h":
        for r in range(ny - bh + 1):
            for c in range(nx - bw):
                L = B[r:r+bh, c:c+bw].reshape(-1)
                R = B[r:r+bh, c+1:c+1+bw].reshape(-1)
                if np.all(L >= 0) and np.all(R >= 0):
                    left_blocks.append(L); right_blocks.append(R)
    elif direction == "v":
        for r in range(ny - bh):
            for c in range(nx - bw + 1):
                L = B[r:r+bh, c:c+bw].reshape(-1)
                R = B[r+1:r+1+bh, c:c+bw].reshape(-1)
                if np.all(L >= 0) and np.all(R >= 0):
                    left_blocks.append(L); right_blocks.append(R)
    else:
        raise ValueError("direction must be 'h' or 'v'")
    if not left_blocks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), bh * bw
    LW, _ = _encode_binary_words(np.asarray(left_blocks, dtype=np.int64))
    RW, _ = _encode_binary_words(np.asarray(right_blocks, dtype=np.int64))
    return LW, RW, bh * bw


def recursive_equalizer_rows(F: np.ndarray, patch_sizes: Iterable[int] = (1, 2, 4)) -> list[dict]:
    """Compute recursive flux equalizers for local patch fields.

    patch size convention:
      1 -> neighbouring single plaquettes;
      2 -> neighbouring adjacent-pair patches, both 1x2 and 2x1;
      4 -> neighbouring 2x2 blocks.
    """
    B = _flux_bits(F)
    rows: list[dict] = []
    for p in patch_sizes:
        p = int(p)
        shapes: list[tuple[int, int, str]] = []
        if p == 1:
            shapes = [(1, 1, "single")]
        elif p == 2:
            shapes = [(1, 2, "pair_h"), (2, 1, "pair_v")]
        elif p == 4:
            shapes = [(2, 2, "block_2x2")]
        else:
            # Fallback: p as a horizontal p-cell block.
            shapes = [(1, p, f"block_1x{p}")]
        for bh, bw, name in shapes:
            for direction in ("h", "v"):
                L, R, word_bits = _adjacent_word_pairs(B, bh, bw, direction)
                row = _equalizer_for_words(L, R, word_bits, f"{name}_adj_{direction}")
                row.update(patch_size=int(p), block_h=int(bh), block_w=int(bw), direction=direction)
                rows.append(row)
    return rows


def _parity(values: np.ndarray) -> int | None:
    V = np.asarray(values, dtype=np.int64)
    V = V[V >= 0]
    if V.size == 0:
        return None
    return int(V.sum() % 2)


def parity_features(F: np.ndarray) -> dict:
    B = _flux_bits(F)
    ny, nx = B.shape
    finite = B >= 0
    vals = B[finite]
    n_neg = int(vals.sum()) if vals.size else 0
    out: dict = dict(
        n_valid_plaquettes=int(vals.size),
        n_total_plaquettes=int(B.size),
        n_negative_flux=int(n_neg),
        negative_fraction=float(n_neg / vals.size) if vals.size else math.nan,
        total_flux_parity=int(n_neg % 2) if vals.size else -1,
    )
    row_parities = []
    for r in range(ny):
        row_parities.append(_parity(B[r, :]))
    col_parities = []
    for c in range(nx):
        col_parities.append(_parity(B[:, c]))
    out.update(
        row_parities=";".join("." if x is None else str(int(x)) for x in row_parities),
        col_parities=";".join("." if x is None else str(int(x)) for x in col_parities),
    )
    # 2x2 block parity map.
    block_parities = []
    if ny >= 2 and nx >= 2:
        for r in range(ny-1):
            row = []
            for c in range(nx-1):
                row.append(_parity(B[r:r+2, c:c+2].reshape(-1)))
            block_parities.append(row)
    out["block2_parity_map"] = "/".join("".join("." if x is None else str(int(x)) for x in row) for row in block_parities)
    if block_parities:
        flat = [x for row in block_parities for x in row if x is not None]
        out["block2_odd_fraction"] = float(np.mean(flat)) if flat else math.nan
    else:
        out["block2_odd_fraction"] = math.nan
    return out


# --------------------------------------------------------------------------- #
# trajectory diagnostic
# --------------------------------------------------------------------------- #

@dataclass
class EdgeLabelMap:
    left: dict[int, int]
    right: dict[int, int]
    n_classes: int


def _fit_edge_label_maps(lat: MicroLattice, S_ref: np.ndarray) -> dict:
    maps: dict = {}
    for key, ed in lat.edges.items():
        L = _encode_words_from_state(S_ref, ed.panel_left, lat.q)
        R = _encode_words_from_state(S_ref, ed.panel_right, lat.q)
        eq = exact_equalizer_labels(L, R, lat.q, lat.w)
        labels = np.asarray(eq["labels"], dtype=np.int64)
        left_map: dict[int, int] = {}
        right_map: dict[int, int] = {}
        for l, r, z in zip(L, R, labels):
            left_map.setdefault(int(l), int(z))
            right_map.setdefault(int(r), int(z))
        maps[key] = EdgeLabelMap(left=left_map, right=right_map, n_classes=int(eq["shared_classes"]))
    return maps


def _state_edge_label(lat: MicroLattice, S: np.ndarray, key, maps: dict) -> np.ndarray:
    ed = lat.edges[key]
    L = _encode_words_from_state(S, ed.panel_left, lat.q)
    R = _encode_words_from_state(S, ed.panel_right, lat.q)
    mp: EdgeLabelMap = maps[key]
    out = np.full(len(S), -1, dtype=np.int64)
    for i, (l, r) in enumerate(zip(L, R)):
        zl = mp.left.get(int(l), -1)
        zr = mp.right.get(int(r), -2)
        if zl >= 0 and zl == zr:
            out[i] = int(zl)
    return out


def state_flux_time_series(
    lat: MicroLattice,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    horizon: int = 16,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute a state-dependent flux-like time series.

    The edge label maps are fit on the one-step reference ensemble.  At each
    later tick we read edge labels through the frozen maps and define each
    plaquette's state-flux as the XOR/parity of its four edge labels when all
    four are binary and valid.  This is a trajectory diagnostic, distinct from
    structural holonomy.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    S = _initial_states(lat, initial_mode, int(n_random_initial), rng)
    S_ref = _step_states(lat.joint, S, lat.schedule)
    maps = _fit_edge_label_maps(lat, S_ref)
    flux_states = []
    valid_fracs = []
    for t in range(int(horizon) + 1):
        if t > 0:
            S = _step_states(lat.joint, S, lat.schedule)
        labels = {key: _state_edge_label(lat, S, key, maps) for key in lat.edges}
        grid_samples = np.full((len(S), lat.ny, lat.nx), -1, dtype=np.int64)
        for r, c in lat.plaquettes:
            keys = [
                _edge_key((r, c), (r, c + 1)),
                _edge_key((r, c), (r + 1, c)),
                _edge_key((r, c + 1), (r + 1, c + 1)),
                _edge_key((r + 1, c), (r + 1, c + 1)),
            ]
            vals = [labels[k] for k in keys]
            ok = np.ones(len(S), dtype=bool)
            for v in vals:
                ok &= (v >= 0) & (v < 2)
            par = np.zeros(len(S), dtype=np.int64)
            for v in vals:
                par ^= np.where(v >= 0, v, 0)
            grid_samples[ok, int(r), int(c)] = par[ok]
        valid_fracs.append(float(np.mean(grid_samples >= 0)))
        # Encode the whole flux configuration for each sample.
        flat = grid_samples.reshape(len(S), -1)
        words = np.full(len(S), -1, dtype=np.int64)
        valid = np.all(flat >= 0, axis=1)
        if np.any(valid) and flat.shape[1] <= 62:
            powers = (2 ** np.arange(flat.shape[1], dtype=np.int64))[None, :]
            words[valid] = (flat[valid] * powers).sum(axis=1)
        flux_states.append(words)
    # Consecutive MI/conservation on full flux words.
    trans = []
    for a, b in zip(flux_states[:-1], flux_states[1:]):
        ok = (a >= 0) & (b >= 0)
        if not np.any(ok):
            trans.append(dict(valid_fraction=0.0, conservation_fraction=math.nan, mi_norm=math.nan, n_states=0))
            continue
        aa, bb = a[ok], b[ok]
        cons = float(np.mean(aa == bb))
        H = _entropy_discrete(bb)
        I = _mutual_info_discrete(aa, bb)
        trans.append(dict(valid_fraction=float(np.mean(ok)), conservation_fraction=cons, mi_norm=float(I / H) if H > 1e-12 else math.nan, n_states=int(len(np.unique(bb)))))
    return dict(valid_fracs=valid_fracs, transitions=trans)


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _entropy_discrete(x: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(x, dtype=np.int64), return_counts=True)
    return _entropy_from_counts(counts)


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    xv, xi = np.unique(x, return_inverse=True)
    yv, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xv), len(yv)), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)
    return float(_entropy_from_counts(joint.sum(axis=1)) + _entropy_from_counts(joint.sum(axis=0)) - _entropy_from_counts(joint.reshape(-1)))


# --------------------------------------------------------------------------- #
# audit driver / analysis / plotting
# --------------------------------------------------------------------------- #

def audit_recursive_flux(
    winner_path: str,
    mode: Mode = "algebraic",
    top_n: int = 50,
    patch_sizes: Iterable[int] = (1, 2, 4),
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    min_transport: float = 0.80,
    min_accuracy: float = 0.95,
    horizon: int = 16,
    measurement_seed: int = 24681357,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    winners = _load_winners(winner_path)[:int(top_n)]
    main_rows: list[dict] = []
    constraint_rows: list[dict] = []
    traj_rows: list[dict] = []
    rng0 = np.random.default_rng(int(measurement_seed))
    for i, w in enumerate(winners):
        lat = _winner_lattice(w)
        rng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
        meas = measure_microscopic_lattice(lat, initial_mode=initial_mode, n_random_initial=int(n_random_initial), min_transport=float(min_transport), min_accuracy=float(min_accuracy), rng=rng)
        F = np.asarray(meas["flux_grid"], dtype=float)
        pf = parity_features(F)
        ds = domain_stats(F)
        star = vertex_star_defect_proxy(F)
        eq_rows = recursive_equalizer_rows(F, patch_sizes=patch_sizes)
        adj_res = [r for r in eq_rows if r["patch_size"] == 1 and r["field"].startswith("single")]
        max_res = max([float(r["residual_bits"]) for r in eq_rows if np.isfinite(r["residual_bits"])], default=math.nan)
        adj_max_res = max([float(r["residual_bits"]) for r in adj_res if np.isfinite(r["residual_bits"])], default=math.nan)
        nontrivial_eq = [r for r in eq_rows if np.isfinite(r["residual_bits"]) and float(r["residual_bits"]) > 1e-9]
        relation_counts = Counter(r["relation_type"] for r in eq_rows)
        row = dict(
            winner_index=int(i),
            fitness=float(w.get("fitness", math.nan)) if isinstance(w, dict) else math.nan,
            nx=int(lat.nx), ny=int(lat.ny), q=int(lat.q), w=int(lat.w),
            z2_fraction=float(meas["z2_fraction"]),
            valid_fraction=float(meas["valid_fraction"]),
            nontrivial_fraction=float(meas["nontrivial_fraction"]),
            flux_map=flux_map_string(F),
            recursive_max_residual_bits=float(max_res) if np.isfinite(max_res) else math.nan,
            adjacency_max_residual_bits=float(adj_max_res) if np.isfinite(adj_max_res) else math.nan,
            n_nontrivial_recursive_equalizers=int(len(nontrivial_eq)),
            relation_type_counts=json.dumps({str(k): int(v) for k, v in relation_counts.items()}),
            **pf, **ds, **star,
            seed_metric_fitness=_metric(w, "fitness", math.nan),
            seed_metric_z2_fraction=_metric(w, "z2_plaquette_fraction", math.nan),
            seed_metric_nontrivial_fraction=_metric(w, "nontrivial_plaquette_fraction", math.nan),
        )
        if mode == "trajectory":
            tr = state_flux_time_series(lat, initial_mode=initial_mode, n_random_initial=int(n_random_initial), horizon=int(horizon), rng=rng)
            vals = tr["valid_fracs"]
            trows = tr["transitions"]
            row.update(
                trajectory_mean_valid_flux_fraction=float(np.nanmean(vals)) if vals else math.nan,
                trajectory_mean_conservation_fraction=float(np.nanmean([x["conservation_fraction"] for x in trows])) if trows else math.nan,
                trajectory_mean_mi_norm=float(np.nanmean([x["mi_norm"] for x in trows])) if trows else math.nan,
                trajectory_max_flux_states=int(max([x["n_states"] for x in trows], default=0)),
            )
            for t, rr in enumerate(trows):
                traj_rows.append(dict(winner_index=int(i), t=int(t), **rr))
        main_rows.append(row)
        for rr in eq_rows:
            rr = dict(rr)
            rr.update(winner_index=int(i), flux_map=flux_map_string(F), nx=int(lat.nx), ny=int(lat.ny))
            constraint_rows.append(rr)
        if verbose:
            print(f"recursive-flux winner {i} done", flush=True)
    df = pd.DataFrame(main_rows)
    cdf = pd.DataFrame(constraint_rows)
    tdf = pd.DataFrame(traj_rows)
    summary = analyze_recursive_flux(df, cdf, tdf, mode=mode)
    summary["winner_path"] = str(winner_path)
    return df, cdf, tdf, summary


def analyze_recursive_flux(df: pd.DataFrame, cdf: pd.DataFrame, tdf: pd.DataFrame | None = None, mode: str = "algebraic") -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_winners=0)
    mean_z2 = float(df["z2_fraction"].mean())
    full_valid = float(np.mean(df["z2_fraction"].to_numpy(dtype=float) >= 1.0 - 1e-12))
    mean_adj = float(df["adjacency_max_residual_bits"].replace([np.inf, -np.inf], np.nan).mean())
    mean_rec = float(df["recursive_max_residual_bits"].replace([np.inf, -np.inf], np.nan).mean())
    nontriv_recursive_frac = float(np.mean(df["n_nontrivial_recursive_equalizers"].to_numpy(dtype=float) > 0))
    star_neg = float(df["interior_star_negative_fraction"].replace([np.inf, -np.inf], np.nan).mean()) if "interior_star_negative_fraction" in df else math.nan
    total_parity_counts = {str(k): int(v) for k, v in Counter(df["total_flux_parity"].astype(int).tolist()).items()} if "total_flux_parity" in df else {}
    flux_map_counts = {str(k): int(v) for k, v in Counter(df["flux_map"].tolist()).most_common(20)} if "flux_map" in df else {}
    # The verdict deliberately distinguishes a global/patch invariant from a localized defect signal.
    if nontriv_recursive_frac > 0.75 and mean_adj > 0.5 and full_valid > 0.8:
        verdict = "RECURSIVE FLUX-CONSTRAINT SIGNAL: selected Z2 flux grids contain nontrivial second-level local equalizers"
    elif nontriv_recursive_frac > 0.25 and full_valid > 0.8:
        verdict = "PARTIAL RECURSIVE FLUX STRUCTURE: some second-level flux constraints/defect proxies are present"
    elif full_valid > 0.8:
        verdict = "FORMAL Z2 FLUX LATTICE: full valid flux maps, but no strong recursive local constraint signal"
    else:
        verdict = "NO STABLE FLUX LATTICE TO AUDIT"
    out = dict(
        verdict=verdict,
        mode=str(mode),
        n_winners=int(len(df)),
        mean_z2_fraction=mean_z2,
        full_valid_fraction=full_valid,
        mean_nontrivial_fraction=float(df["nontrivial_fraction"].mean()) if "nontrivial_fraction" in df else math.nan,
        mean_recursive_max_residual_bits=mean_rec,
        mean_adjacency_max_residual_bits=mean_adj,
        nontrivial_recursive_equalizer_fraction=nontriv_recursive_frac,
        mean_n_nontrivial_recursive_equalizers=float(df["n_nontrivial_recursive_equalizers"].mean()) if "n_nontrivial_recursive_equalizers" in df else math.nan,
        total_flux_parity_counts=total_parity_counts,
        unique_flux_map_count=int(df["flux_map"].nunique()) if "flux_map" in df else 0,
        dominant_flux_map_fraction=float(df["flux_map"].value_counts(normalize=True).iloc[0]) if "flux_map" in df and len(df) else math.nan,
        top_flux_maps=flux_map_counts,
        mean_domain_wall_fraction=float(df["domain_wall_fraction"].replace([np.inf, -np.inf], np.nan).mean()) if "domain_wall_fraction" in df else math.nan,
        mean_interior_star_negative_fraction=star_neg,
    )
    if cdf is not None and not cdf.empty:
        eq_by_field = []
        for fld, g in cdf.groupby("field"):
            eq_by_field.append(dict(
                field=str(fld),
                mean_residual_bits=float(g["residual_bits"].replace([np.inf, -np.inf], np.nan).mean()),
                nontrivial_fraction=float(np.mean(g["residual_bits"].to_numpy(dtype=float) > 1e-9)),
                mean_shared_classes=float(g["shared_classes"].replace([np.inf, -np.inf], np.nan).mean()),
                relation_type_counts={str(k): int(v) for k, v in Counter(g["relation_type"].astype(str).tolist()).items()},
            ))
        out["equalizers_by_field"] = eq_by_field
    if mode == "trajectory" and tdf is not None and not tdf.empty:
        out.update(
            trajectory_mean_valid_fraction=float(tdf["valid_fraction"].replace([np.inf, -np.inf], np.nan).mean()),
            trajectory_mean_conservation_fraction=float(tdf["conservation_fraction"].replace([np.inf, -np.inf], np.nan).mean()),
            trajectory_mean_mi_norm=float(tdf["mi_norm"].replace([np.inf, -np.inf], np.nan).mean()),
        )
    return out


def plot_recursive_flux(df: pd.DataFrame, cdf: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    ax[0].hist(df["recursive_max_residual_bits"].dropna(), bins=10)
    ax[0].set_title("max recursive residual")
    ax[0].set_xlabel("bits")
    ax[0].set_ylabel("winners")
    if "n_negative_flux" in df:
        ax[1].hist(df["n_negative_flux"].dropna(), bins=range(0, int(df["n_total_plaquettes"].max()) + 2))
        ax[1].set_title("nontrivial plaquette count")
        ax[1].set_xlabel("# F=-1")
    if cdf is not None and not cdf.empty:
        g = cdf.groupby("field")["residual_bits"].mean().sort_values(ascending=False)
        ax[2].bar(range(len(g)), g.values)
        ax[2].set_xticks(range(len(g)))
        ax[2].set_xticklabels(g.index, rotation=60, ha="right", fontsize=7)
        ax[2].set_title("recursive equalizers by field")
        ax[2].set_ylabel("mean residual bits")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(',') if str(x).strip()]


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recursive flux-constraint audit on selected Z2 micro-lattice winners.")
    p.add_argument("winner_path")
    p.add_argument("--mode", choices=["algebraic", "trajectory"], default="algebraic")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--patch-sizes", default="1,2,4")
    p.add_argument("--initial-mode", choices=["source_all", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--min-transport", type=float, default=0.80)
    p.add_argument("--min-accuracy", type=float, default=0.95)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--measurement-seed", type=int, default=24681357)
    p.add_argument("--out", default="example_results/recursive_flux_audit.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    df, cdf, tdf, summary = audit_recursive_flux(
        args.winner_path,
        mode=args.mode,
        top_n=int(args.top_n),
        patch_sizes=_parse_ints(args.patch_sizes),
        initial_mode=args.initial_mode,
        n_random_initial=int(args.n_random_initial),
        min_transport=float(args.min_transport),
        min_accuracy=float(args.min_accuracy),
        horizon=int(args.horizon),
        measurement_seed=int(args.measurement_seed),
        verbose=not bool(args.quiet),
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        base = os.path.splitext(args.out)[0]
        df.to_csv(args.out, index=False)
        cdf.to_csv(base + "_constraints.csv", index=False)
        if not tdf.empty:
            tdf.to_csv(base + "_trajectory.csv", index=False)
        with open(base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or '.', exist_ok=True)
        plot_recursive_flux(df, cdf, args.plot)
    print(json.dumps(summary, indent=2))
    if args.out:
        print(f"wrote {args.out}")
        print(f"wrote {os.path.splitext(args.out)[0]}_constraints.csv")
    if args.plot:
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
