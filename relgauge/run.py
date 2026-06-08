#!/usr/bin/env python3
"""
run.py -- one-command experiment driver.

Usage:
    python -m relgauge.run --preset quick   --out results_quick
    python -m relgauge.run --preset full    --out results_full
    python -m relgauge.run --config my.json --out results_custom

Writes to <out>/:
    instances.csv     one row per random system (raw data)
    aggregated.csv    bootstrap-aggregated metrics per cell
    config.json       exact grid + seeds + provenance (reproducibility)
    report.md         the written falsification verdict
    fig_*.png         figures
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import core as C
from . import experiments as E
from . import observables as obs
from . import report as R


PRESETS = {
    # fast smoke test (~1-2 min): proves the pipeline + reproduces theorems
    "quick": dict(
        grid=dict(ensemble=["cycle", "scc"], k=[2, 3, 4], q=[2, 4, 6, 8],
                  w=[1], density=[2]),
        n_instances=60, passive_horizons=[1, 2], do_active=True,
        do_partition=True, n_boot=800,
        structural=dict(k=4, q=3, n=60, fusion_kP=3, fusion_q=3, fusion_r=2,
                        fusion_events=40)),
    # publication run (hours): wider k, q; multiple ensembles & widths
    "full": dict(
        grid=dict(ensemble=["cycle", "scc", "regular"],
                  k=[2, 3, 4, 5, 6], q=[2, 3, 4, 6, 8, 12, 16],
                  w=[1, 2, 3], density=[1, 2, 3]),
        n_instances=400, passive_horizons=[1, 2, 3], do_active=True,
        do_partition=True, n_boot=2000,
        structural=dict(k=5, q=4, n=300, fusion_kP=3, fusion_q=4, fusion_r=2,
                        fusion_events=200)),
}


def run_structural(cfg, base_seed=0):
    rng = np.random.default_rng(base_seed + 999)
    k, q, n = cfg["k"], cfg["q"], cfg["n"]
    # S1 + S2
    s1 = 0
    sync = 0
    asyncc = 0
    for _ in range(n):
        sysm = C.make_random_scc(k, q, 2, rng)
        adj_graph = {v: set() for v in range(k)}
        for v in range(k):
            for p in sysm.preds[v]:
                adj_graph[p].add(v)
        if C.condensation_is_dag(adj_graph, range(k)):
            s1 += 1
        if obs.check_light_cone(sysm, 60, rng, "sync"):
            sync += 1
        if obs.check_light_cone(sysm, 60, rng, "async"):
            asyncc += 1
    # S3
    ok = 0
    viol = 0
    ne = cfg["fusion_events"]
    for _ in range(ne):
        _, ge = obs.fusion_monotonicity_event(
            cfg["fusion_kP"], cfg["fusion_q"], cfg["fusion_r"], rng, feedback=False)
        ok += int(ge)
    for _ in range(ne):
        _, ge = obs.fusion_monotonicity_event(
            cfg["fusion_kP"], cfg["fusion_q"], cfg["fusion_r"], rng, feedback=True)
        viol += int(not ge)
    return dict(S1_dag=(s1 == n), S1_n=n,
                S2_sync=sync / n, S2_async=asyncc / n,
                S3_ok=ok, S3_n=ne, S3_viol=viol, S3_viol_n=ne)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="quick")
    ap.add_argument("--config", type=str, default=None,
                    help="JSON file overriding the preset")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    os.makedirs(args.out, exist_ok=True)
    print(f"[relgauge] preset={args.preset} -> {args.out}")

    print("[1/4] structural checks ...")
    structural = run_structural(cfg["structural"], base_seed=args.seed)
    print("      ", {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in structural.items()})

    print("[2/4] ensemble sweep ...")
    df = E.run_sweep(cfg["grid"], n_instances=cfg["n_instances"],
                     base_seed=args.seed,
                     passive_horizons=tuple(cfg["passive_horizons"]),
                     do_active=cfg["do_active"], do_partition=cfg["do_partition"])
    df.to_csv(f"{args.out}/instances.csv", index=False)

    print("[3/4] aggregation ...")
    agg_frames = []
    metrics = [m for m in R.RESOLVER_METRICS if m in df] + \
              [c for c in ["part_surv_image_count", "part_surv_full_count",
                           "orbit_dim_image", "orbit_dim_recurrent"] if c in df]
    for m in metrics:
        agg_frames.append(E.aggregate(df, m, n_boot=cfg["n_boot"], seed=args.seed))
    agg = pd.concat(agg_frames, ignore_index=True)
    agg.to_csv(f"{args.out}/aggregated.csv", index=False)

    print("[4/4] plots + report ...")
    R.make_plots(df, args.out)
    prov = E.provenance()
    res = R.write_report(df, args.out, prov, structural)

    with open(f"{args.out}/config.json", "w") as f:
        json.dump(dict(preset=args.preset, seed=args.seed, grid=cfg["grid"],
                       n_instances=cfg["n_instances"], structural=cfg["structural"],
                       provenance=prov, overall_verdict=res["overall"]),
                  f, indent=2)

    print("\n===== VERDICT =====")
    for m, c in res["overall"].items():
        print(f"  {m:16s}: {c}")
    print(f"\nWrote {args.out}/report.md and figures.")


if __name__ == "__main__":
    main()
