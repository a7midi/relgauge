"""
experiments.py -- ensemble sweeps over (ensemble, k, q, w, density) with full
reproducibility (seed-per-instance), producing tidy per-instance and aggregated
tables.

The sweep is the experimental apparatus; the scientific verdict is assembled in
report.py from these tables.
"""
from __future__ import annotations

import platform
import sys
import time

import numpy as np
import pandas as pd

from . import core as C
from . import observables as obs
from . import stats as st


def _make_system(ensemble, k, q, rng, density):
    if ensemble == "cycle":
        return C.make_cycle(k, q, rng)
    if ensemble == "scc":
        return C.make_random_scc(k, q, density, rng)
    if ensemble == "regular":
        return C.make_regular_scc(k, q, density, rng)
    raise ValueError(ensemble)


def run_sweep(grid: dict, n_instances: int, base_seed: int = 0,
              passive_horizons=(1, 2), do_active=True, do_partition=True,
              verbose=True) -> pd.DataFrame:
    """grid: dict with keys among {'ensemble','k','q','w','density'} mapping to
    lists of values; the full Cartesian product is swept.  Each cell uses
    `n_instances` random systems with deterministic per-instance seeds derived
    from (base_seed, cell, instance) for exact reproducibility.

    Returns a tidy DataFrame, one row per instance.
    """
    ensembles = grid.get("ensemble", ["cycle"])
    ks = grid.get("k", [3])
    qs = grid.get("q", [8])
    ws = grid.get("w", [1])
    densities = grid.get("density", [2])

    rows = []
    cells = [(e, k, q, w, d) for e in ensembles for k in ks for q in qs
             for w in ws for d in densities]
    t0 = time.time()
    for ci, (e, k, q, w, d) in enumerate(cells):
        if w >= k:                       # boundary must be a proper subset
            continue
        if e == "cycle" and d != densities[0]:
            continue                     # density irrelevant for cycle; avoid dup
        for inst in range(n_instances):
            seed = (base_seed * 1_000_003 + ci * 10_007 + inst) % (2 ** 32)
            rng = np.random.default_rng(seed)
            sysm = _make_system(e, k, q, rng, d)
            m = obs.measure_instance(sysm, w, passive_horizons=passive_horizons,
                                     do_active=do_active, do_partition=do_partition)
            m.update(dict(ensemble=e, density=(0 if e == "cycle" else d),
                          instance=inst, seed=seed))
            rows.append(m)
        if verbose:
            print(f"  cell {ci+1}/{len(cells)}  "
                  f"ens={e} k={k} q={q} w={w} d={d}  "
                  f"[{time.time()-t0:.1f}s]", flush=True)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, metric: str, by=("ensemble", "k", "q", "w"),
              n_boot=1000, seed=0) -> pd.DataFrame:
    """Bootstrap-aggregate a metric over instances within each cell."""
    rng = np.random.default_rng(seed)
    out = []
    for key, sub in df.groupby(list(by)):
        mean, lo, hi, n = st.bootstrap_ci(sub[metric].values, n_boot=n_boot, rng=rng)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(dict(metric=metric, mean=mean, lo=lo, hi=hi, n=n))
        if "ratio" in sub:
            rec["ratio"] = float(sub["ratio"].iloc[0])
        out.append(rec)
    return pd.DataFrame(out)


def provenance() -> dict:
    return dict(python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, pandas=pd.__version__,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
