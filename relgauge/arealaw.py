"""
arealaw.py -- does the boundary-resolvable information obey an AREA LAW?

For a single SCC with boundary of width w, the *absolute* resolvable
information is
    I(w, k) = log2 #{ [x]~ : x in Rec(S) }
where ~ is the boundary observational equivalence (passive finite-horizon by
default; the physical observer is passive).  We test:

  AREA LAW    : I grows ~ linearly in the boundary width w, with a slope alpha
                that is independent of the bulk size k (I saturates in k).
  VOLUME LAW  : I grows with k (I ~ log2|Rec|); the boundary is transparent.

This is the edge-mode / entanglement-entropy reframing.  The intensive
invariant -- if it exists -- is the area-law coefficient alpha (information per
boundary vertex), which REPLACES the dead survival fraction.

Empirically (this conversation): cycles are transparent (volume); the open
question is whether DENSE (opaque) random SCCs are area-law.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import core as C
from . import quotients as Q


def resolvable_info(sys: C.RelationalSystem, w: int, sm=None,
                    mode="passive", horizon=1) -> dict:
    """Absolute resolvable information through a width-w boundary, on Rec."""
    if sm is None:
        sm = sys.all_step_maps()
    adj = C.orbit_adjacency_fast(sm)
    rec = C.recurrent_states(adj)
    boundary = tuple(range(min(w, sys.k)))
    out = dict(k=sys.k, q=sys.q, w=w, n_rec=int(len(rec)))
    if len(rec) < 2:
        out.update(I_bits=np.nan, logRec_bits=np.nan, opacity=np.nan)
        return out
    if mode == "active":
        block = Q.boundary_bisimulation(sys, boundary, sm)
    else:
        block = Q.passive_signature_classes(sys, boundary, horizon, sm)
    ncl = len(set(block[rec].tolist()))
    I_bits = np.log2(ncl)
    logRec = np.log2(len(rec))
    out.update(I_bits=float(I_bits), logRec_bits=float(logRec),
               opacity=float(I_bits / logRec))   # 1 = transparent, ~0 = opaque
    return out


def run_area_law(ensemble="scc", ks=(3, 4, 5), q=4, density_fn=None,
                 n_instances=80, mode="passive", horizon=1, base_seed=0,
                 verbose=True) -> pd.DataFrame:
    """Sweep (k, w) for a chosen ensemble at fixed alphabet q.

    density_fn(k) -> number of extra internal edges (default: ~1 extra per
    vertex, a genuinely *dense* family so the opaque regime is probed and the
    average degree is held roughly constant as k grows).
    """
    if density_fn is None:
        density_fn = lambda k: k                      # ~2 in-edges/vertex
    rows = []
    for ki, k in enumerate(ks):
        for w in range(1, k):                         # proper boundaries
            for inst in range(n_instances):
                seed = (base_seed * 7919 + ki * 131 + w * 17 + inst) % 2**32
                rng = np.random.default_rng(seed)
                extra = int(density_fn(k))
                if ensemble == "scc":
                    sysm = C.make_random_scc(k, q, extra, rng)
                elif ensemble == "regular":
                    sysm = C.make_regular_scc(k, q, max(2, 1 + extra // k), rng)
                else:
                    sysm = C.make_cycle(k, q, rng)
                m = resolvable_info(sysm, w, mode=mode, horizon=horizon)
                m.update(ensemble=ensemble, density=extra, seed=seed)
                rows.append(m)
        if verbose:
            print(f"  area-law k={k} done", flush=True)
    return pd.DataFrame(rows)


def analyze_area_law(df: pd.DataFrame) -> dict:
    """Fit I ~ slope*w within each k; test slope-constancy across k
    (area-law) and saturation of I in k at fixed w."""
    res = dict(slopes_by_k={}, saturation={}, opacity_at_w1={})
    for k in sorted(df.k.unique()):
        sub = df[df.k == k]
        g = sub.groupby("w")["I_bits"].mean().dropna()
        if len(g) >= 2:
            slope, intercept = np.polyfit(g.index.values, g.values, 1)
            res["slopes_by_k"][int(k)] = float(slope)
        s = sub[sub.w == 1]
        if len(s):
            res["opacity_at_w1"][int(k)] = float(s.opacity.mean())
    # saturation at w=1: I_bits vs k
    s1 = df[df.w == 1].groupby("k")["I_bits"].mean().dropna()
    sL = df[df.w == 1].groupby("k")["logRec_bits"].mean().dropna()
    res["saturation"] = dict(
        I_bits_by_k={int(k): float(v) for k, v in s1.items()},
        logRec_by_k={int(k): float(v) for k, v in sL.items()})
    # verdict: the discriminating signal is whether the resolvable info
    # SATURATES in the bulk (area) or tracks the volume (transparent).  Use the
    # maximal/active resolver for this; the w-slope is reported but NOT gated on
    # (it is confounded: when w->k the boundary equals the bulk).
    slopes = list(res["slopes_by_k"].values())
    res["slope_spread"] = (float(max(slopes) - min(slopes))
                           if len(slopes) >= 2 else None)
    Ivals = list(s1.values)
    Lvals = list(sL.values)
    I_growth = (Ivals[-1] - Ivals[0]) if len(Ivals) >= 2 else np.nan
    L_growth = (Lvals[-1] - Lvals[0]) if len(Lvals) >= 2 else np.nan
    opac = list(res["opacity_at_w1"].values())
    opac_last = opac[-1] if opac else np.nan
    if np.isnan(I_growth) or np.isnan(L_growth) or abs(L_growth) < 1e-9:
        verdict = "NO DATA / INCONCLUSIVE"
    elif I_growth / L_growth > 0.6:
        verdict = ("VOLUME-LAW / transparent (I tracks bulk; the boundary "
                   "reveals the whole state given enough observation -- NO "
                   "holographic hiding)")
    elif I_growth / L_growth < 0.25 and opac_last < 0.5:
        verdict = ("AREA-LAW (I saturates in bulk; boundary reveals a "
                   "boundary-bounded amount -- holographic hiding)")
    else:
        verdict = "INCONCLUSIVE (weak/partial saturation; larger k needed)"
    res["I_growth_over_volume_growth"] = (
        float(I_growth / L_growth) if abs(L_growth) > 1e-9 else None)
    res["verdict"] = verdict
    return res


def plot_area_law(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in sorted(df.k.unique()):
        g = df[df.k == k].groupby("w")["I_bits"].mean().dropna()
        ax[0].plot(g.index, g.values, "o-", label=f"k={k}")
    ax[0].set_xlabel("boundary width w"); ax[0].set_ylabel(r"$I$ (bits)")
    ax[0].set_title("resolvable info vs boundary (area-law => linear, k-stacked)")
    ax[0].legend(fontsize=8)
    s1 = df[df.w == 1].groupby("k")
    ax[1].plot(list(s1.groups), [s1.get_group(k).I_bits.mean() for k in s1.groups],
               "o-", label="I(w=1)")
    ax[1].plot(list(s1.groups), [s1.get_group(k).logRec_bits.mean() for k in s1.groups],
               "s--", label=r"$\log_2|\mathrm{Rec}|$ (volume)")
    ax[1].set_xlabel("bulk size k"); ax[1].set_ylabel("bits")
    ax[1].set_title("saturation in bulk (area => I flat, volume => I rises)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    import sys as _sys
    q = int(_sys.argv[1]) if len(_sys.argv) > 1 else 4
    df = run_area_law(ensemble="scc", ks=(3, 4, 5), q=q, n_instances=80, mode="active")
    res = analyze_area_law(df)
    print("\nslopes by k:", res["slopes_by_k"])
    print("opacity (I/logRec) at w=1:", res["opacity_at_w1"])
    print("saturation:", res["saturation"])
    print("VERDICT:", res["verdict"])
