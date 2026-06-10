"""
wilsonlattice.py -- effective multi-plaquette Z2 lattice / Wilson-loop test.

This module is the next scale-up after shared-corner square holonomy.  The
shared-square experiments produce a finite C2 ~= Z2 holonomy sector: each valid
plaquette has a path holonomy Delta that is either identity (+1 flux) or the
nontrivial C2 element (-1 flux).  This module builds an effective two-dimensional
Z2 plaquette lattice from those observed plaquette fluxes and measures Wilson
loops.

Important scope
---------------
This is an EFFECTIVE plaquette-lattice test, not yet a fully microscopic grid of
coupled relational vertices.  It answers:

    If the selected shared-square C2 plaquettes are used as local Z2 fluxes on
    a 2D lattice, do Wilson loops show area-law, perimeter-law, or flat behavior?

In two-dimensional Z2 gauge language, if plaquette fluxes are independent with
P(F=-1)=p, then for a rectangular loop C enclosing area A,

    <W(C)> = (1 - 2p)^A,

so |<W>| decays with area.  A link-noise control instead produces perimeter-law
scaling.  The module supports both modes.

Recommended run, using blind shared-square selected C2 data:

python -m relgauge.wilsonlattice example_results/blind_shared_square_q4_w1.csv ^
  --mode empirical_plaquette ^
  --lattice-sizes 6,8,10,12 ^
  --n-lattices 500 ^
  --max-loop-side 5 ^
  --out example_results/wilson_lattice_q4_z2.csv ^
  --plot example_results/fig_wilson_lattice_q4_z2.png

Useful controls:

python -m relgauge.wilsonlattice --mode flat --lattice-sizes 8 --out example_results/wilson_flat.csv
python -m relgauge.wilsonlattice --mode iid_link --p-flux 0.15 --lattice-sizes 8,12 --out example_results/wilson_link_noise.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

Mode = Literal["empirical_plaquette", "iid_plaquette", "iid_link", "flat"]


# --------------------------------------------------------------------------- #
# CSV -> C2 flux distribution
# --------------------------------------------------------------------------- #
def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_c2_flux_distribution(
    csv_path: str,
    group_filter: str = "C2",
    valid_only: bool = True,
    identity_names: Iterable[str] = ("identity",),
    nontrivial_names: Iterable[str] = ("cycle_2", "cycle_2_1", "cycle_2_1_1", "cycle_2_2"),
) -> dict:
    """Load an empirical Z2 plaquette-flux distribution from square data.

    For the blind shared-square C2 sector, `delta_type == identity` is +1 and
    `delta_type == cycle_2` is -1.  The broader nontrivial_names are accepted so
    the function can be used on controls; when group_filter=C2 only cycle_2 is
    expected.
    """
    if not csv_path:
        raise ValueError("csv_path is required for empirical_plaquette mode")
    df = pd.read_csv(csv_path)
    work = df.copy()
    if valid_only and "valid_square" in work.columns:
        work = work[_as_bool_series(work["valid_square"])]
    if group_filter and "generated_group_name" in work.columns:
        work = work[work["generated_group_name"].astype(str) == str(group_filter)]
    if "delta_type" not in work.columns:
        raise ValueError("CSV must contain a delta_type column")
    identity_names = set(str(x) for x in identity_names)
    nontrivial_names = set(str(x) for x in nontrivial_names)
    dt = work["delta_type"].astype(str)
    n_identity = int(dt.isin(identity_names).sum())
    n_nontrivial = int(dt.isin(nontrivial_names).sum())
    n_used = n_identity + n_nontrivial
    if n_used <= 0:
        raise ValueError("no identity/nontrivial C2 plaquettes found after filtering")
    p = n_nontrivial / n_used
    return dict(
        csv_path=str(csv_path),
        group_filter=str(group_filter),
        n_rows_total=int(len(df)),
        n_rows_after_filter=int(len(work)),
        n_identity=n_identity,
        n_nontrivial=n_nontrivial,
        n_used=n_used,
        p_nontrivial=float(p),
        mean_flux=float(1.0 - 2.0 * p),
        delta_type_counts={str(k): int(v) for k, v in dt.value_counts().to_dict().items()},
    )


# --------------------------------------------------------------------------- #
# Lattice generation and Wilson loops
# --------------------------------------------------------------------------- #
def _prefix_sum_int(A: np.ndarray) -> np.ndarray:
    P = np.zeros((A.shape[0] + 1, A.shape[1] + 1), dtype=np.int64)
    P[1:, 1:] = A.cumsum(axis=0).cumsum(axis=1)
    return P


def _rect_sum(P: np.ndarray, x: int, y: int, a: int, b: int) -> int:
    return int(P[x + a, y + b] - P[x, y + b] - P[x + a, y] + P[x, y])


def _plaquette_lattice(Lx: int, Ly: int, p_nontrivial: float, rng: np.random.Generator) -> np.ndarray:
    # Boolean flux: True means nontrivial C2 flux (-1), False means identity (+1).
    return rng.random((int(Lx), int(Ly))) < float(p_nontrivial)


def _link_lattice(Lx: int, Ly: int, p_link_flip: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Horizontal edges h[x,y] from (x,y) to (x+1,y), shape Lx x (Ly+1).
    # Vertical edges v[x,y] from (x,y) to (x,y+1), shape (Lx+1) x Ly.
    h = rng.random((int(Lx), int(Ly) + 1)) < float(p_link_flip)
    v = rng.random((int(Lx) + 1, int(Ly))) < float(p_link_flip)
    return h, v


def _wilson_from_plaquettes(flux_neg: np.ndarray, a: int, b: int) -> np.ndarray:
    Lx, Ly = flux_neg.shape
    P = _prefix_sum_int(flux_neg.astype(np.int64))
    vals = []
    for x in range(0, Lx - int(a) + 1):
        for y in range(0, Ly - int(b) + 1):
            neg = _rect_sum(P, x, y, int(a), int(b))
            vals.append(-1 if (neg % 2) else 1)
    return np.asarray(vals, dtype=np.int8)


def _wilson_from_links(h_neg: np.ndarray, v_neg: np.ndarray, a: int, b: int) -> np.ndarray:
    Lx, Ly1 = h_neg.shape
    Ly = Ly1 - 1
    vals = []
    # brute force is fine for the small diagnostic lattices used here
    for x in range(0, Lx - int(a) + 1):
        for y in range(0, Ly - int(b) + 1):
            neg = 0
            # bottom/top horizontal edges
            neg += int(h_neg[x:x + int(a), y].sum())
            neg += int(h_neg[x:x + int(a), y + int(b)].sum())
            # left/right vertical edges
            neg += int(v_neg[x, y:y + int(b)].sum())
            neg += int(v_neg[x + int(a), y:y + int(b)].sum())
            vals.append(-1 if (neg % 2) else 1)
    return np.asarray(vals, dtype=np.int8)


def measure_wilson_lattice(
    Lx: int,
    Ly: int,
    mode: Mode,
    p: float,
    n_lattices: int,
    max_loop_side: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    Lx = int(Lx); Ly = int(Ly)
    max_a = min(int(max_loop_side), Lx)
    max_b = min(int(max_loop_side), Ly)
    buckets: dict[tuple[int, int], list[int]] = {(a, b): [] for a in range(1, max_a + 1) for b in range(1, max_b + 1)}
    for _ in range(int(n_lattices)):
        if mode in ("empirical_plaquette", "iid_plaquette", "flat"):
            flux = _plaquette_lattice(Lx, Ly, 0.0 if mode == "flat" else float(p), rng)
            for a, b in buckets:
                buckets[(a, b)].extend(_wilson_from_plaquettes(flux, a, b).tolist())
        elif mode == "iid_link":
            h, v = _link_lattice(Lx, Ly, float(p), rng)
            for a, b in buckets:
                buckets[(a, b)].extend(_wilson_from_links(h, v, a, b).tolist())
        else:
            raise ValueError(f"unknown mode {mode!r}")
    for (a, b), vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        mean_w = float(arr.mean()) if len(arr) else math.nan
        rows.append(dict(
            Lx=Lx, Ly=Ly, mode=str(mode), p=float(p), n_lattices=int(n_lattices),
            loop_w=int(a), loop_h=int(b), area=int(a * b), perimeter=int(2 * (a + b)),
            n_loops=int(len(arr)), mean_W=mean_w, abs_mean_W=float(abs(mean_w)),
            frac_negative=float((arr < 0).mean()) if len(arr) else math.nan,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Fitting / analysis
# --------------------------------------------------------------------------- #
def _linear_fit(x: np.ndarray, y: np.ndarray, k_params: int = 2) -> dict:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2:
        return dict(slope=math.nan, intercept=math.nan, rss=math.nan, aic=math.inf, r2=math.nan, n=int(len(x)))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    resid = y - pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 1e-12 else 1.0
    eps = 1e-12
    aic = float(len(y) * math.log(max(rss / max(len(y), 1), eps)) + 2 * int(k_params))
    return dict(slope=float(slope), intercept=float(intercept), rss=rss, aic=aic, r2=r2, n=int(len(x)))


def analyze_wilson(df: pd.DataFrame, flux_info: dict | None = None) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    # Use one row per area/perimeter by averaging across lattice sizes.  This avoids
    # overweighting the larger lattices in the regression.
    g = df.groupby(["area", "perimeter"], as_index=False).agg(mean_W=("mean_W", "mean"), abs_mean_W=("abs_mean_W", "mean"), n_loops=("n_loops", "sum"))
    eps = 1e-12
    y = np.log(np.maximum(g["abs_mean_W"].to_numpy(float), eps))
    area_fit = _linear_fit(g["area"].to_numpy(float), y)
    per_fit = _linear_fit(g["perimeter"].to_numpy(float), y)
    area_alpha = float(-area_fit["slope"]) if math.isfinite(area_fit["slope"]) else math.nan
    per_beta = float(-per_fit["slope"]) if math.isfinite(per_fit["slope"]) else math.nan
    p = float(df["p"].iloc[0]) if "p" in df.columns and len(df) else math.nan
    mode = str(df["mode"].iloc[0]) if "mode" in df.columns and len(df) else "unknown"
    m = float(1.0 - 2.0 * p) if math.isfinite(p) else math.nan
    expected_area_alpha = float(-math.log(abs(m))) if math.isfinite(m) and abs(m) > 1e-12 and abs(m) < 1.0 else (0.0 if abs(m) == 1.0 else math.inf)
    expected_perimeter_beta = expected_area_alpha
    aic_gap = float(per_fit["aic"] - area_fit["aic"]) if math.isfinite(per_fit["aic"]) and math.isfinite(area_fit["aic"]) else math.nan
    mean_abs_w = float(df["abs_mean_W"].mean())

    # Direct generative-model residuals.  This is the primary diagnostic because
    # area and perimeter are highly collinear for small rectangular loops.
    pred_area = np.asarray([m ** int(a) for a in g["area"]], dtype=float) if math.isfinite(m) else np.full(len(g), math.nan)
    pred_per = np.asarray([m ** int(p0) for p0 in g["perimeter"]], dtype=float) if math.isfinite(m) else np.full(len(g), math.nan)
    obs = g["mean_W"].to_numpy(float)
    area_model_rmse = float(np.sqrt(np.nanmean((obs - pred_area) ** 2))) if len(obs) else math.nan
    perimeter_model_rmse = float(np.sqrt(np.nanmean((obs - pred_per) ** 2))) if len(obs) else math.nan

    if mean_abs_w > 0.98:
        verdict = "FLAT Z2 LATTICE: Wilson loops are nearly identity"
    elif mode in ("empirical_plaquette", "iid_plaquette"):
        if area_model_rmse < 0.06 or (math.isfinite(area_fit["r2"]) and area_fit["r2"] > 0.85):
            verdict = "EFFECTIVE Z2 AREA-LAW: plaquette fluxes give Wilson loops governed by enclosed area"
        else:
            verdict = "PLAQUETTE-LAW MODEL WITH NOISY FINITE-SAMPLE SCALING"
    elif mode == "iid_link":
        if perimeter_model_rmse < 0.06 or (math.isfinite(per_fit["r2"]) and per_fit["r2"] > 0.85):
            verdict = "PERIMETER-LAW CONTROL: link noise gives Wilson loops governed by boundary length"
        else:
            verdict = "LINK-NOISE MODEL WITH NOISY FINITE-SAMPLE SCALING"
    elif math.isfinite(aic_gap) and aic_gap > 2.0:
        verdict = "EFFECTIVE Z2 AREA-LAW: Wilson loops decay primarily with enclosed plaquette area"
    elif math.isfinite(aic_gap) and aic_gap < -2.0:
        verdict = "PERIMETER-LAW CONTROL: Wilson loops decay primarily with boundary length"
    else:
        verdict = "MIXED / INCONCLUSIVE WILSON SCALING"
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        mode=mode,
        p=float(p),
        mean_flux=m,
        expected_area_alpha=expected_area_alpha,
        expected_perimeter_beta=expected_perimeter_beta,
        mean_abs_W=mean_abs_w,
        area_fit=area_fit,
        perimeter_fit=per_fit,
        area_alpha=area_alpha,
        perimeter_beta=per_beta,
        aic_gap_perimeter_minus_area=aic_gap,
        area_model_rmse=area_model_rmse,
        perimeter_model_rmse=perimeter_model_rmse,
        flux_info=flux_info or {},
        by_area={str(int(k)): float(v) for k, v in df.groupby("area")["mean_W"].mean().to_dict().items()},
        by_area_abs={str(int(k)): float(v) for k, v in df.groupby("area")["abs_mean_W"].mean().to_dict().items()},
    )


def run_wilson_test(
    csv_path: str | None = None,
    mode: Mode = "empirical_plaquette",
    p_flux: float | None = None,
    lattice_sizes: Iterable[int] = (6, 8, 10, 12),
    n_lattices: int = 500,
    max_loop_side: int = 5,
    base_seed: int = 0,
    group_filter: str = "C2",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    flux_info: dict | None = None
    if mode == "empirical_plaquette":
        if csv_path is None:
            raise ValueError("csv_path is required for empirical_plaquette mode")
        flux_info = load_c2_flux_distribution(csv_path, group_filter=group_filter)
        p = float(flux_info["p_nontrivial"])
    elif mode == "flat":
        p = 0.0 if p_flux is None else float(p_flux)
    else:
        if p_flux is None:
            raise ValueError("--p-flux is required for iid_plaquette or iid_link mode")
        p = float(p_flux)
    rows = []
    for L in lattice_sizes:
        seed = (int(base_seed) * 1000003 + int(L) * 9176 + int(round(p * 100000)) * 31 + hash(str(mode)) % 1000) % 2**32
        rng = np.random.default_rng(seed)
        d = measure_wilson_lattice(int(L), int(L), mode=mode, p=p, n_lattices=int(n_lattices), max_loop_side=int(max_loop_side), rng=rng)
        d["seed"] = int(seed)
        rows.append(d)
        if verbose:
            print(f"wilson lattice L={L} mode={mode} done", flush=True)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = analyze_wilson(df, flux_info=flux_info)
    return df, summary


# --------------------------------------------------------------------------- #
# Plotting / CLI
# --------------------------------------------------------------------------- #
def plot_wilson(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    g_area = df.groupby("area", as_index=False).agg(mean_W=("mean_W", "mean"), abs_mean_W=("abs_mean_W", "mean"))
    ax[0].plot(g_area["area"], g_area["mean_W"], "o-", label="signed")
    ax[0].plot(g_area["area"], g_area["abs_mean_W"], "s--", label="absolute")
    ax[0].set_xlabel("area enclosed")
    ax[0].set_ylabel(r"$\langle W\rangle$")
    ax[0].set_title("Wilson loop vs area")
    ax[0].legend(fontsize=8)
    eps = 1e-12
    ax[1].plot(g_area["area"], np.log(np.maximum(g_area["abs_mean_W"], eps)), "o-")
    ax[1].set_xlabel("area")
    ax[1].set_ylabel(r"$\log |\langle W\rangle|$")
    ax[1].set_title("area-law diagnostic")
    g_per = df.groupby("perimeter", as_index=False).agg(abs_mean_W=("abs_mean_W", "mean"))
    ax[2].plot(g_per["perimeter"], np.log(np.maximum(g_per["abs_mean_W"], eps)), "o-")
    ax[2].set_xlabel("perimeter")
    ax[2].set_ylabel(r"$\log |\langle W\rangle|$")
    ax[2].set_title("perimeter-law diagnostic")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Effective multi-plaquette Z2 lattice / Wilson-loop test.")
    p.add_argument("csv", nargs="?", default=None, help="CSV from blindsharedsquare/sharedsquareholonomy; required for empirical_plaquette mode")
    p.add_argument("--mode", choices=["empirical_plaquette", "iid_plaquette", "iid_link", "flat"], default="empirical_plaquette")
    p.add_argument("--p-flux", type=float, default=None, help="Nontrivial plaquette probability, or link flip probability in iid_link mode")
    p.add_argument("--lattice-sizes", default="6,8,10,12")
    p.add_argument("--n-lattices", type=int, default=500)
    p.add_argument("--max-loop-side", type=int, default=5)
    p.add_argument("--group-filter", default="C2")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", default="example_results/wilson_lattice.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    df, summary = run_wilson_test(
        csv_path=args.csv,
        mode=args.mode,
        p_flux=args.p_flux,
        lattice_sizes=_parse_ints(args.lattice_sizes),
        n_lattices=int(args.n_lattices),
        max_loop_side=int(args.max_loop_side),
        base_seed=int(args.base_seed),
        group_filter=str(args.group_filter),
        verbose=not bool(args.quiet),
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_wilson(df, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
