"""
c2entropyaudit.py -- is the first live transported square quotient entropically binary?

This audit is a characterization/diagnostic, not a selector.  It estimates the
quotient-size distribution for shared-corner square candidates under several
sources:

  * random null samples;
  * saved blind shared-square winners;
  * local mutation neighborhoods of saved winners.

For each candidate it measures the live/bijective square with the same post-hoc
shared-square evaluator and records the minimum transported alphabet size |S|
across the four edges.  The main statistic is

    P(|S| = 2 | valid live transported square)

and the competing probabilities for |S| >= 3 and |S| = q.

This is designed to test the theorem-candidate suggested by the project notes:
among random/near-random finite deterministic rule tables conditioned on live
transport, the cheapest nontrivial quotient is overwhelmingly binary.

Recommended run
---------------
python -m relgauge.c2entropyaudit 4 ^
  --w 1 ^
  --random-samples 1000 ^
  --winner-paths example_results/blind_shared_square_winners_q4_w1.pkl ^
  --top-n 50 ^
  --neighborhood-samples 2000 ^
  --mutation-rates 0.02,0.05,0.10 ^
  --out example_results/c2_entropy_audit_q4_w1.csv ^
  --plot example_results/fig_c2_entropy_audit_q4_w1.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from . import blindsharedsquare as BS
from . import sharedsquareholonomy as SH


def _parse_floats(s: str) -> list[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def _safe_float(x, default=math.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _min_classes(row: dict) -> int:
    keys = ["exact_B_classes", "exact_C_classes", "exact_Dtop_classes", "exact_Dbottom_classes"]
    vals = [int(row.get(k, 0) or 0) for k in keys]
    return int(min(vals)) if vals else 0


def _row_from_candidate(
    sq: SH.SharedSquare,
    rng: np.random.Generator,
    source: str,
    q: int,
    w: int,
    initial_mode: str,
    n_random_initial: int,
    generation: int = 0,
    candidate: int = 0,
    mutation_rate: float = 0.0,
) -> dict:
    r = BS.evaluate_candidate(
        sq,
        rng,
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        fitness="transport",
        generation=int(generation),
        pretrain_generations=0,
    )
    m = _min_classes(r)
    r.update(
        audit_source=source,
        q=int(q), w=int(w),
        min_classes=int(m),
        class_residual_qcoords=float(math.log(m, q)) if m > 0 else 0.0,
        binary_classes=bool(m == 2),
        ternary_or_more=bool(m >= 3),
        full_classes=bool(m == int(q) ** int(w)),
        mutation_rate=float(mutation_rate),
        generation=int(generation),
        candidate=int(candidate),
    )
    return r


def _load_winners(paths: Iterable[str], top_n: int | None = None, require_valid: bool = False):
    out = []
    for path in paths:
        if not path:
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        winners = data.get("winners", []) if isinstance(data, dict) else data
        for i, item in enumerate(winners):
            if not isinstance(item, dict) or "square" not in item:
                continue
            metrics = dict(item.get("metrics", {}))
            if require_valid and not bool(metrics.get("valid_square", False)):
                continue
            fit = _safe_float(item.get("fitness", metrics.get("blind_fitness", 0.0)), 0.0)
            out.append((path, i, fit, item["square"], metrics))
    out.sort(key=lambda t: (
        bool(t[4].get("valid_square", False)),
        bool(t[4].get("nontrivial_path_holonomy", False)),
        float(t[4].get("min_residual_qcoords", 0.0) or 0.0),
        float(t[2] or 0.0),
    ), reverse=True)
    if top_n is not None and int(top_n) > 0:
        out = out[:int(top_n)]
    return out


def run_c2_entropy_audit(
    q: int,
    w: int,
    random_samples: int = 1000,
    winner_paths: list[str] | None = None,
    top_n: int = 50,
    neighborhood_samples: int = 0,
    mutation_rates: list[float] | None = None,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    base_seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    q = int(q); w = int(w)
    rows: list[dict] = []
    # Random null samples.
    for i in range(int(random_samples)):
        seed = (int(base_seed) * 1000003 + q * 10007 + w * 1009 + i) % 2**32
        rng = np.random.default_rng(seed)
        sq = BS.random_square(q, w, rng)
        r = _row_from_candidate(sq, rng, "random", q, w, initial_mode, n_random_initial, generation=0, candidate=i)
        r.update(seed=int(seed), winner_path="")
        rows.append(r)
    if verbose and random_samples:
        print(f"c2-entropy random samples={random_samples} done", flush=True)

    winners = _load_winners(winner_paths or [], top_n=top_n, require_valid=False)
    for j, (path, idx, fit, sq, metrics) in enumerate(winners):
        rng = np.random.default_rng((int(base_seed) * 65537 + j * 17 + idx) % 2**32)
        r = _row_from_candidate(sq, rng, "winner", q, w, initial_mode, n_random_initial, generation=0, candidate=j)
        r.update(seed=math.nan, winner_path=str(path), winner_index=int(idx), saved_fitness=float(fit))
        rows.append(r)
    if verbose and winners:
        print(f"c2-entropy winners={len(winners)} done", flush=True)

    if neighborhood_samples and winners:
        rates = mutation_rates or [0.02, 0.05, 0.10]
        for rr, rate in enumerate(rates):
            for i in range(int(neighborhood_samples)):
                seed = (int(base_seed) * 1000003 + rr * 100003 + i * 37 + q * 101) % 2**32
                rng = np.random.default_rng(seed)
                _, _, _, parent, _ = winners[int(rng.integers(0, len(winners)))]
                child = BS.mutate_square(
                    parent,
                    rng,
                    entry_rate=float(rate),
                    table_rate=float(rate),
                    target="interface",
                    force_one=True,
                )
                r = _row_from_candidate(child, rng, "neighborhood", q, w, initial_mode, n_random_initial, generation=0, candidate=i, mutation_rate=float(rate))
                r.update(seed=int(seed), winner_path="neighborhood")
                rows.append(r)
            if verbose:
                print(f"c2-entropy neighborhood mu={rate} samples={neighborhood_samples} done", flush=True)
    return pd.DataFrame(rows)


def _conditional_stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return dict(n=0, valid_count=0, valid_fraction=0.0, binary_given_valid=math.nan, ge3_given_valid=math.nan, full_given_valid=math.nan)
    valid = sub[sub.get("valid_square", False).astype(bool)] if "valid_square" in sub else sub.iloc[0:0]
    out = dict(
        n=int(len(sub)),
        valid_count=int(len(valid)),
        valid_fraction=float(len(valid) / len(sub)) if len(sub) else 0.0,
    )
    if len(valid):
        out.update(
            binary_given_valid=float(valid["binary_classes"].astype(bool).mean()),
            ge3_given_valid=float(valid["ternary_or_more"].astype(bool).mean()),
            full_given_valid=float(valid["full_classes"].astype(bool).mean()),
            min_classes_counts={str(int(k)): int(v) for k, v in Counter(valid["min_classes"].fillna(0).astype(int)).items()},
            group_counts={str(k): int(v) for k, v in Counter(valid.get("generated_group_name", pd.Series([], dtype=str)).astype(str)).items()},
            delta_type_counts={str(k): int(v) for k, v in Counter(valid.get("delta_type", pd.Series([], dtype=str)).astype(str)).items()},
        )
    else:
        out.update(binary_given_valid=math.nan, ge3_given_valid=math.nan, full_given_valid=math.nan, min_classes_counts={}, group_counts={}, delta_type_counts={})
    return out


def analyze_c2_entropy(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO C2 ENTROPY DATA", n_rows=0)
    by_source = {str(src): _conditional_stats(sub) for src, sub in df.groupby("audit_source")}
    all_stats = _conditional_stats(df)
    # Determine verdict by the conditional valid pool, weighting only actually
    # live/bijective squares.  Random nulls may have no valid pool at this size.
    b = all_stats.get("binary_given_valid", math.nan)
    ge3 = all_stats.get("ge3_given_valid", math.nan)
    if not math.isnan(b) and b >= 0.90:
        verdict = "C2 ENTROPY SIGNAL: conditioned live transported squares are overwhelmingly binary"
    elif not math.isnan(b) and b > 0.50:
        verdict = "PARTIAL C2 ENTROPY SIGNAL: binary quotients dominate the conditioned valid pool"
    elif all_stats.get("valid_count", 0) == 0:
        verdict = "NO VALID CONDITIONED POOL: random samples did not yield live transported squares"
    else:
        verdict = "NO C2 ENTROPY DOMINANCE: conditioned valid pool is not predominantly binary"
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        overall=all_stats,
        by_source=by_source,
    )


def plot_c2_entropy(df: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        for ax in axes: ax.set_axis_off()
    else:
        valid = df[df["valid_square"].astype(bool)] if "valid_square" in df else df.iloc[0:0]
        if len(valid):
            # Counts of min classes by source.
            pivot = valid.pivot_table(index="min_classes", columns="audit_source", values="q", aggfunc="count", fill_value=0).sort_index()
            pivot.plot(kind="bar", ax=axes[0])
            axes[0].set_title("valid conditioned |S| counts")
            axes[0].set_xlabel("min |S| over edges")
            # Binary fraction by source.
            bf = valid.groupby("audit_source")["binary_classes"].mean().sort_index()
            axes[1].bar(bf.index.astype(str), bf.values)
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title("P(|S|=2 | valid)")
            axes[1].tick_params(axis="x", rotation=30)
            # Valid fraction by source.
            vf = df.groupby("audit_source")["valid_square"].mean().sort_index()
            axes[2].bar(vf.index.astype(str), vf.values)
            axes[2].set_ylim(0, 1.05)
            axes[2].set_title("valid live square fraction")
            axes[2].tick_params(axis="x", rotation=30)
        else:
            counts = df.get("min_classes", pd.Series([], dtype=int)).fillna(0).astype(int).value_counts().sort_index()
            axes[0].bar([str(x) for x in counts.index], counts.values)
            axes[0].set_title("all candidate min |S|")
            axes[1].set_axis_off(); axes[2].set_axis_off()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit whether conditioned live transported squares are entropically binary.")
    ap.add_argument("q", type=int)
    ap.add_argument("--w", type=int, default=1)
    ap.add_argument("--random-samples", type=int, default=1000)
    ap.add_argument("--winner-paths", default="", help="comma-separated saved blind shared-square winner pickle paths")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--neighborhood-samples", type=int, default=0)
    ap.add_argument("--mutation-rates", default="0.02,0.05,0.10")
    ap.add_argument("--initial-mode", default="joint_random", choices=["source_all", "source_random", "joint_random"])
    ap.add_argument("--n-random-initial", type=int, default=4096)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="example_results/c2_entropy_audit.csv")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)
    paths = [p for p in str(args.winner_paths).split(",") if p.strip()]
    df = run_c2_entropy_audit(
        int(args.q), int(args.w),
        random_samples=int(args.random_samples),
        winner_paths=paths,
        top_n=int(args.top_n),
        neighborhood_samples=int(args.neighborhood_samples),
        mutation_rates=_parse_floats(args.mutation_rates),
        initial_mode=str(args.initial_mode),
        n_random_initial=int(args.n_random_initial),
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = analyze_c2_entropy(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    if args.plot:
        plot_c2_entropy(df, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
