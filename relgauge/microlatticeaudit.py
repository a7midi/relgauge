"""
microlatticeaudit.py -- audit selected microscopic Z2 lattice winners.

This module is a diagnostic for the winners produced by blindmicrolattice.py.
It does not select or optimize.  It loads saved microscopic coupled-lattice
winners and measures the gauge-field observables that are already present:

  * plaquette flux maps F_p in {+1,-1};
  * Wilson loops W(gamma) = product_{p inside gamma} F_p on the actual
    selected coupled lattice;
  * flux-flux correlations by plaquette Manhattan distance;
  * nontrivial-flux domain statistics;
  * a local star-parity / defect proxy (product of incident plaquette fluxes
    around a lattice vertex; this is a flux diagnostic, not a hand-inserted
    matter charge).

The key distinction from wilsonlattice.py is that the input is not an
independent plaquette sample.  The input is a pickled list of microscopic
lattice systems whose neighbouring plaquettes share edge transporters.

Example
-------
python -m relgauge.microlatticeaudit example_results/blind_microlattice_winners_q4_3x3.pkl ^
  --top-n 50 ^
  --initial-mode joint_random ^
  --n-random-initial 4096 ^
  --out example_results/microlattice_audit_q4_3x3.csv ^
  --plot example_results/fig_microlattice_audit_q4_3x3.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter, defaultdict, deque
from typing import Iterable

import numpy as np
import pandas as pd

from .microlattice import MicroLattice, measure_microscopic_lattice


def _flux_char(x: float) -> str:
    if not np.isfinite(x):
        return "."
    return "+" if float(x) > 0 else "-"


def flux_map_string(F: np.ndarray) -> str:
    """Compact human-readable flux map: '+' for identity, '-' for C2 flux."""
    F = np.asarray(F, dtype=float)
    rows = []
    for r in range(F.shape[0]):
        rows.append("".join(_flux_char(x) for x in F[r]))
    return "/".join(rows)


def _finite_flux_values(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return F[np.isfinite(F)]


def wilson_values(F: np.ndarray, height: int, width: int) -> list[float]:
    """Wilson-loop values from actual selected plaquette flux grid."""
    F = np.asarray(F, dtype=float)
    h = int(height); w = int(width)
    ny, nx = F.shape
    vals: list[float] = []
    if h <= 0 or w <= 0 or h > ny or w > nx:
        return vals
    for r in range(0, ny - h + 1):
        for c in range(0, nx - w + 1):
            sub = F[r:r+h, c:c+w]
            if np.all(np.isfinite(sub)):
                vals.append(float(np.prod(sub)))
    return vals


def all_wilson_rows(F: np.ndarray, max_loop_side: int = 4) -> list[dict]:
    F = np.asarray(F, dtype=float)
    ny, nx = F.shape
    rows: list[dict] = []
    for h in range(1, min(int(max_loop_side), ny) + 1):
        for w in range(1, min(int(max_loop_side), nx) + 1):
            vals = wilson_values(F, h, w)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            rows.append(dict(
                loop_h=int(h), loop_w=int(w), area=int(h*w), perimeter=int(2*(h+w)),
                n_loops=int(len(arr)), W_mean=float(arr.mean()), W_abs_mean=float(abs(arr.mean())),
                W_values=";".join(str(int(x)) for x in arr.tolist()),
            ))
    return rows


def flux_correlation_rows(F: np.ndarray, max_distance: int | None = None) -> list[dict]:
    """Flux product correlations by Manhattan distance between plaquettes."""
    F = np.asarray(F, dtype=float)
    ny, nx = F.shape
    vals = _finite_flux_values(F)
    mean_flux = float(vals.mean()) if vals.size else math.nan
    maxd = int(max_distance) if max_distance is not None else (ny + nx - 2)
    rows: list[dict] = []
    for d in range(1, maxd + 1):
        products = []
        for r1 in range(ny):
            for c1 in range(nx):
                if not np.isfinite(F[r1, c1]):
                    continue
                for r2 in range(ny):
                    for c2 in range(nx):
                        if (r2, c2) <= (r1, c1):
                            continue
                        if abs(r1-r2) + abs(c1-c2) != d:
                            continue
                        if np.isfinite(F[r2, c2]):
                            products.append(float(F[r1,c1] * F[r2,c2]))
        if products:
            arr = np.asarray(products, dtype=float)
            rows.append(dict(
                distance=int(d), n_pairs=int(len(arr)), corr_raw=float(arr.mean()),
                corr_connected=float(arr.mean() - mean_flux*mean_flux) if np.isfinite(mean_flux) else math.nan,
                mean_flux=float(mean_flux),
            ))
    return rows


def _component_sizes(mask: np.ndarray) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r in range(ny):
        for c in range(nx):
            if seen[r,c] or not mask[r,c]:
                continue
            q = deque([(r,c)]); seen[r,c] = True; n = 0
            while q:
                i,j = q.popleft(); n += 1
                for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    ii,jj = i+di, j+dj
                    if 0 <= ii < ny and 0 <= jj < nx and (not seen[ii,jj]) and mask[ii,jj]:
                        seen[ii,jj] = True; q.append((ii,jj))
            sizes.append(int(n))
    return sizes


def domain_stats(F: np.ndarray) -> dict:
    """Connected-domain and domain-wall diagnostics for +/- plaquette flux."""
    F = np.asarray(F, dtype=float)
    finite = np.isfinite(F)
    neg = finite & (F < 0)
    pos = finite & (F > 0)
    neg_sizes = _component_sizes(neg)
    pos_sizes = _component_sizes(pos)
    wall = 0
    valid_adj = 0
    ny, nx = F.shape
    for r in range(ny):
        for c in range(nx):
            if not finite[r,c]:
                continue
            for di,dj in ((1,0),(0,1)):
                rr, cc = r+di, c+dj
                if rr < ny and cc < nx and finite[rr,cc]:
                    valid_adj += 1
                    if F[r,c] != F[rr,cc]:
                        wall += 1
    return dict(
        neg_domain_count=int(len(neg_sizes)),
        neg_domain_sizes=";".join(map(str, sorted(neg_sizes, reverse=True))),
        neg_domain_max=int(max(neg_sizes) if neg_sizes else 0),
        pos_domain_count=int(len(pos_sizes)),
        pos_domain_sizes=";".join(map(str, sorted(pos_sizes, reverse=True))),
        pos_domain_max=int(max(pos_sizes) if pos_sizes else 0),
        domain_wall_edges=int(wall),
        valid_adjacencies=int(valid_adj),
        domain_wall_fraction=float(wall / valid_adj) if valid_adj else math.nan,
    )


def vertex_star_defect_proxy(F: np.ndarray) -> dict:
    """Flux-star parity around lattice vertices.

    For each lattice vertex, multiply all adjacent finite plaquette fluxes.  On
    interior vertices of a complete 2D grid this is a four-plaquette local
    parity.  We report the fraction with negative parity.  This is a diagnostic
    for local flux organization; it is not a matter charge inserted by hand.
    """
    F = np.asarray(F, dtype=float)
    ny, nx = F.shape
    vals_all = []
    vals_int = []
    for vr in range(ny + 1):
        for vc in range(nx + 1):
            adj = []
            for dr, dc in ((-1,-1),(-1,0),(0,-1),(0,0)):
                pr, pc = vr + dr, vc + dc
                if 0 <= pr < ny and 0 <= pc < nx and np.isfinite(F[pr,pc]):
                    adj.append(float(F[pr,pc]))
            if adj:
                prod = float(np.prod(adj))
                vals_all.append(prod)
                if 0 < vr < ny and 0 < vc < nx and len(adj) == 4:
                    vals_int.append(prod)
    def frac_neg(vals):
        if not vals:
            return math.nan
        return float(np.mean([x < 0 for x in vals]))
    return dict(
        vertex_star_count=int(len(vals_all)),
        vertex_star_negative_fraction=frac_neg(vals_all),
        interior_star_count=int(len(vals_int)),
        interior_star_negative_fraction=frac_neg(vals_int),
    )


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
    raise ValueError("winner entry lacks 'lattice' field")


def _safe_metric(w: dict, key: str, default=math.nan):
    try:
        return w.get("metrics", {}).get(key, default)
    except Exception:
        return default


def audit_winners(
    winner_path: str,
    top_n: int = 50,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    max_loop_side: int = 3,
    min_transport: float = 0.80,
    min_accuracy: float = 0.95,
    measurement_seed: int = 24680,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    winners = _load_winners(winner_path)
    # Keep stored order, which is already ranked by the producer.  Top_n <= 0 => all.
    if int(top_n) > 0:
        winners = winners[:int(top_n)]
    winner_rows: list[dict] = []
    loop_rows: list[dict] = []
    corr_rows: list[dict] = []
    plaquette_rows: list[dict] = []
    for idx, w in enumerate(winners):
        lat = _winner_lattice(w)
        rng = np.random.default_rng((int(measurement_seed) + 1009 * int(idx)) % (2**32))
        meas = measure_microscopic_lattice(
            lat,
            initial_mode=initial_mode,  # type: ignore[arg-type]
            n_random_initial=int(n_random_initial),
            min_transport=float(min_transport),
            min_accuracy=float(min_accuracy),
            rng=rng,
        )
        F = np.asarray(meas["flux_grid"], dtype=float)
        finite = _finite_flux_values(F)
        z2_fraction = float(np.isfinite(F).mean()) if F.size else math.nan
        nontriv_fraction = float(np.mean(finite < 0)) if finite.size else 0.0
        mean_flux = float(finite.mean()) if finite.size else math.nan
        dom = domain_stats(F)
        star = vertex_star_defect_proxy(F)
        base = dict(
            winner_index=int(idx),
            stored_fitness=float(w.get("fitness", math.nan)) if isinstance(w, dict) else math.nan,
            stored_selected=bool(_safe_metric(w, "selected_lattice", False)),
            stored_full_valid=bool(_safe_metric(w, "full_valid_lattice", False)),
            q=int(getattr(lat, "q", -1)), w=int(getattr(lat, "w", -1)), nx=int(getattr(lat, "nx", -1)), ny=int(getattr(lat, "ny", -1)),
            valid_fraction=float(meas.get("valid_fraction", math.nan)),
            z2_fraction=float(meas.get("z2_fraction", z2_fraction)),
            nontrivial_fraction=float(meas.get("nontrivial_fraction", nontriv_fraction)),
            measured_z2_fraction=z2_fraction,
            measured_nontrivial_fraction=nontriv_fraction,
            mean_flux=mean_flux,
            flux_map=flux_map_string(F),
            group_counts=json.dumps(meas.get("group_counts", {}), sort_keys=True),
            delta_counts=json.dumps(meas.get("delta_counts", {}), sort_keys=True),
        )
        row = {**base, **dom, **star}
        winner_rows.append(row)
        # Per-plaquette rows.
        for p in meas.get("plaquettes", []):
            pr = dict(winner_index=int(idx))
            for k, v in p.items():
                if isinstance(v, (np.integer, np.floating)):
                    v = v.item()
                pr[k] = v
            plaquette_rows.append(pr)
        # Wilson rows.
        for lr in all_wilson_rows(F, max_loop_side=max_loop_side):
            lr.update(winner_index=int(idx))
            loop_rows.append(lr)
        # Correlation rows.
        for cr in flux_correlation_rows(F):
            cr.update(winner_index=int(idx))
            corr_rows.append(cr)
        if verbose:
            print(f"audit winner {idx}: z2={z2_fraction:.3f}, nontriv={nontriv_fraction:.3f}", flush=True)
    dfw = pd.DataFrame(winner_rows)
    dfl = pd.DataFrame(loop_rows)
    dfc = pd.DataFrame(corr_rows)
    dfp = pd.DataFrame(plaquette_rows)
    summary = summarize_audit(dfw, dfl, dfc, dfp, winner_path=winner_path)
    return dfw, dfl, dfc, dfp, summary


def _group_mean(df: pd.DataFrame, by: str, val: str) -> dict:
    if df.empty or by not in df or val not in df:
        return {}
    g = df.groupby(by)[val].mean().dropna()
    return {str(k): float(v) for k, v in g.items()}


def summarize_audit(dfw: pd.DataFrame, dfl: pd.DataFrame, dfc: pd.DataFrame, dfp: pd.DataFrame, winner_path: str = "") -> dict:
    if dfw.empty:
        return dict(verdict="NO WINNERS", n_winners=0, winner_path=winner_path)
    full_valid_fraction = float(np.mean(dfw["z2_fraction"] >= 0.999)) if "z2_fraction" in dfw else math.nan
    nontriv_any_fraction = float(np.mean(dfw["nontrivial_fraction"] > 0)) if "nontrivial_fraction" in dfw else math.nan
    mean_z2 = float(dfw["z2_fraction"].mean())
    mean_nontriv = float(dfw["nontrivial_fraction"].mean())
    # Wilson loop aggregate by area/perimeter.
    wilson_area = _group_mean(dfl, "area", "W_mean")
    wilson_abs_area = _group_mean(dfl, "area", "W_abs_mean")
    corr_by_distance = _group_mean(dfc, "distance", "corr_raw")
    conn_corr_by_distance = _group_mean(dfc, "distance", "corr_connected")
    # Interpret verdict conservatively.
    if mean_z2 > 0.95 and nontriv_any_fraction > 0.5:
        verdict = "COUPLED Z2 MICRO-LATTICE AUDIT: selected winners contain full valid plaquette flux maps with nontrivial sectors"
    elif mean_z2 > 0.5:
        verdict = "PARTIAL Z2 MICRO-LATTICE AUDIT: selected winners contain substantial valid plaquette flux structure"
    else:
        verdict = "WEAK MICRO-LATTICE AUDIT: selected winners have limited valid Z2 plaquette structure under this measurement"
    best_idx = int(dfw.sort_values(["z2_fraction", "nontrivial_fraction", "stored_fitness"], ascending=False).iloc[0]["winner_index"])
    return dict(
        verdict=verdict,
        winner_path=winner_path,
        n_winners=int(len(dfw)),
        mean_z2_fraction=mean_z2,
        full_valid_fraction=full_valid_fraction,
        mean_nontrivial_fraction=mean_nontriv,
        nontrivial_any_fraction=nontriv_any_fraction,
        mean_flux=float(dfw["mean_flux"].mean()) if "mean_flux" in dfw else math.nan,
        mean_domain_wall_fraction=float(dfw["domain_wall_fraction"].mean()) if "domain_wall_fraction" in dfw else math.nan,
        mean_neg_domain_count=float(dfw["neg_domain_count"].mean()) if "neg_domain_count" in dfw else math.nan,
        mean_neg_domain_max=float(dfw["neg_domain_max"].mean()) if "neg_domain_max" in dfw else math.nan,
        mean_interior_star_negative_fraction=float(dfw["interior_star_negative_fraction"].mean()) if "interior_star_negative_fraction" in dfw else math.nan,
        wilson_mean_by_area=wilson_area,
        wilson_abs_mean_by_area=wilson_abs_area,
        flux_corr_by_distance=corr_by_distance,
        flux_connected_corr_by_distance=conn_corr_by_distance,
        best_winner_index=best_idx,
        best_flux_map=str(dfw[dfw.winner_index == best_idx].iloc[0]["flux_map"]),
    )


def plot_audit(dfw: pd.DataFrame, dfl: pd.DataFrame, dfc: pd.DataFrame, path: str, max_maps: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    if dfw.empty:
        fig.savefig(path); plt.close(fig); return
    # Panel 1: flux-map contact sheet for top winners.
    top = dfw.sort_values(["z2_fraction", "nontrivial_fraction", "stored_fitness"], ascending=False).head(int(max_maps))
    maps = []
    for s in top["flux_map"].tolist():
        rows = str(s).split("/")
        arr = np.zeros((len(rows), max(len(r) for r in rows)), dtype=float)
        arr[:] = np.nan
        for i, rr in enumerate(rows):
            for j, ch in enumerate(rr):
                arr[i,j] = 1 if ch == "+" else (-1 if ch == "-" else np.nan)
        maps.append(arr)
    if maps:
        # Stack horizontally with NaN separators.
        H = max(m.shape[0] for m in maps); W = sum(m.shape[1] for m in maps) + len(maps)-1
        sheet = np.full((H, W), np.nan)
        off = 0
        for m in maps:
            sheet[:m.shape[0], off:off+m.shape[1]] = m
            off += m.shape[1] + 1
        ax[0].imshow(sheet, vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
        ax[0].set_title("top flux maps (+/-)")
        ax[0].set_xticks([]); ax[0].set_yticks([])
    # Panel 2: Wilson |<W>| by area.
    if not dfl.empty:
        g = dfl.groupby("area")["W_abs_mean"].mean().dropna()
        if len(g):
            ax[1].plot(g.index, g.values, "o-")
    ax[1].set_title("actual Wilson |<W>| by area")
    ax[1].set_xlabel("area"); ax[1].set_ylabel("|<W>|")
    # Panel 3: flux correlations.
    if not dfc.empty:
        g = dfc.groupby("distance")["corr_raw"].mean().dropna()
        if len(g):
            ax[2].plot(g.index, g.values, "o-", label="raw")
        gc = dfc.groupby("distance")["corr_connected"].mean().dropna()
        if len(gc):
            ax[2].plot(gc.index, gc.values, "s--", label="connected")
        ax[2].legend(fontsize=8)
    ax[2].set_title("flux correlations")
    ax[2].set_xlabel("Manhattan distance")
    # Panel 4: domain wall vs flux density.
    ax[3].scatter(dfw["nontrivial_fraction"], dfw["domain_wall_fraction"], alpha=0.8)
    ax[3].set_xlabel("nontrivial flux fraction")
    ax[3].set_ylabel("domain-wall fraction")
    ax[3].set_title("flux domains")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Audit selected microscopic Z2 lattice winners.")
    p.add_argument("winner_file", help="Pickle produced by blindmicrolattice --save-winners")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--initial-mode", choices=["source_all", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--max-loop-side", type=int, default=3)
    p.add_argument("--min-transport", type=float, default=0.80)
    p.add_argument("--min-accuracy", type=float, default=0.95)
    p.add_argument("--measurement-seed", type=int, default=24680)
    p.add_argument("--out", default="example_results/microlattice_audit.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    dfw, dfl, dfc, dfp, summary = audit_winners(
        args.winner_file,
        top_n=int(args.top_n), initial_mode=args.initial_mode,
        n_random_initial=int(args.n_random_initial), max_loop_side=int(args.max_loop_side),
        min_transport=float(args.min_transport), min_accuracy=float(args.min_accuracy),
        measurement_seed=int(args.measurement_seed), verbose=not bool(args.quiet),
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        root, ext = os.path.splitext(args.out)
        dfw.to_csv(args.out, index=False)
        dfl.to_csv(root + "_loops.csv", index=False)
        dfc.to_csv(root + "_corr.csv", index=False)
        dfp.to_csv(root + "_plaquettes.csv", index=False)
        with open(root + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_audit(dfw, dfl, dfc, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
