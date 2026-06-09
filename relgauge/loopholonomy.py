"""
loopholonomy.py -- finite holonomy audit for transported interface labels.

This is the stage after ``transportalgebra.py``.

Single temporal-chain winners give live transported label maps

    phi : S1 -> S2.

A loop experiment composes such transported maps around a closed chain:

    Phi_loop = phi_L ... phi_2 phi_1.

For a genuine finite gauge-like structure, the raw loop product is basis-
dependent, but its conjugacy class is invariant under local relabelings at the
vertices.  This module therefore tests two things:

1. Nontriviality: does the transported sector produce non-identity loop
   holonomies?
2. Gauge covariance: under local vertex relabelings

       phi_i -> g_{i+1} phi_i g_i^{-1}

   with g_L = g_0, does the loop product transform as

       Phi_loop -> g_0 Phi_loop g_0^{-1},

   preserving the conjugacy class?

Recommended use
---------------
First save transported winners, e.g. from seeded temporal recovery:

python -m relgauge.seededtemporalrecovery 4 ^
  --ks-B 2 --ks-M 2 --ks-A 2 --ws 1 ^
  --seed-ensembles copy,permutive,block,canalizing,random ^
  --seed-mutation-rates 0.05,0.1 ^
  --runs 5 --population 20 --generations 60 ^
  --fitness live_sum --target interface --proposal-mode mixed ^
  --entry-rate 0.04 --table-rate 0.03 ^
  --initial-mode joint_random --n-random-initial 4096 ^
  --save-winners example_results/seeded_temporal_winners_q4_w1.pkl ^
  --save-winner-top-n 50 ^
  --out example_results/seeded_temporal_recovery_q4_w1_saved.csv

Then run:

python -m relgauge.loopholonomy example_results/seeded_temporal_winners_q4_w1.pkl ^
  --top-n 50 --initial-mode joint_random --n-random-initial 4096 ^
  --min-transport 0.8 --min-source 0.8 --min-accuracy 0.95 ^
  --loop-length 4 --use-unique ^
  --out example_results/loop_holonomy_q4_w1.csv ^
  --plot example_results/fig_loop_holonomy_q4_w1.png

Interpretation
--------------
For the observed q=4,w=1 transported sector, ``transportalgebra.py`` found a
nonabelian order-6 group acting on |S|=3 labels.  This module asks whether
closed products in that sector produce nontrivial conjugacy classes (identity,
transposition, 3-cycle for S3) and whether those classes behave covariantly
under local relabeling.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import transportalgebra as A

InitialMode = A.InitialMode


# --------------------------------------------------------------------------- #
# Permutation helpers
# --------------------------------------------------------------------------- #
def compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """Return p after q: i -> p[q[i]]."""
    return tuple(int(p[int(q[i])]) for i in range(len(p)))


def identity(n: int) -> tuple[int, ...]:
    return tuple(range(int(n)))


def inverse(p: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[int(j)] = int(i)
    return tuple(inv)


def cycle_lengths(p: tuple[int, ...]) -> tuple[int, ...]:
    return A._cycle_lengths(tuple(p))  # type: ignore[attr-defined]


def perm_order(p: tuple[int, ...]) -> int:
    return A._perm_order(tuple(p))  # type: ignore[attr-defined]


def perm_type(p: tuple[int, ...] | None) -> str:
    return A.classify_permutation(p)


def conjugate(g: tuple[int, ...], h: tuple[int, ...]) -> tuple[int, ...]:
    """g h g^{-1}."""
    return compose(g, compose(h, inverse(g)))


def conjugacy_class_in_group(h: tuple[int, ...], group: Iterable[tuple[int, ...]]) -> set[tuple[int, ...]]:
    return {conjugate(g, h) for g in group}


def holonomy_product(edges: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    """Compose edge maps around a loop.

    If edges = [phi_0, phi_1, ..., phi_{L-1}], then the loop map from the
    starting label back to itself is

        phi_{L-1} ... phi_1 phi_0.
    """
    edges = [tuple(e) for e in edges]
    if not edges:
        raise ValueError("at least one edge map required")
    n = len(edges[0])
    h = identity(n)
    for e in edges:
        if len(e) != n:
            raise ValueError("all edge maps must have the same label cardinality")
        h = compose(e, h)
    return h


# --------------------------------------------------------------------------- #
# Extract transported maps from saved winners
# --------------------------------------------------------------------------- #
def extract_transport_permutations(
    winner_path: str,
    top_n: int = 50,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    min_transport: float = 0.80,
    min_source: float = 0.80,
    min_accuracy: float = 0.95,
    min_residual: float = 1e-9,
    base_seed: int = 0,
) -> tuple[list[dict], dict]:
    """Load temporal-chain winners and return selected bijective label maps."""
    df, summary = A.audit_transport_algebra(
        winner_path,
        top_n=int(top_n),
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        source_for_liveness=source_for_liveness,
        min_transport=float(min_transport),
        min_source=float(min_source),
        min_accuracy=float(min_accuracy),
        min_residual=float(min_residual),
        base_seed=int(base_seed),
    )
    maps: list[dict] = []
    for _, row in df.iterrows():
        if not bool(row.get("selected_for_group", False)):
            continue
        try:
            p_raw = json.loads(str(row.get("permutation_json", "null")))
        except Exception:
            p_raw = None
        if p_raw is None:
            continue
        p = tuple(int(x) for x in p_raw)
        if sorted(p) != list(range(len(p))):
            continue
        maps.append(
            dict(
                winner_index=int(row.get("winner_index", -1)),
                seed_ensemble=str(row.get("seed_ensemble", "unknown")),
                seed_mutation_rate=float(row.get("seed_mutation_rate", math.nan)),
                q=int(row.get("q", 0)),
                w=int(row.get("w", 0)),
                n=int(len(p)),
                permutation=p,
                permutation_type=perm_type(p),
                transport=float(row.get("transport_mi_over_Hs2", math.nan)),
                source=float(row.get("source_to_s2_mi_over_Hs2", math.nan)),
                accuracy=float(row.get("best_accuracy", math.nan)),
                s1_residual=float(row.get("s1_residual_qcoords", math.nan)),
                s2_residual=float(row.get("s2_residual_qcoords", math.nan)),
            )
        )
    return maps, summary


# --------------------------------------------------------------------------- #
# Loop enumeration and gauge covariance
# --------------------------------------------------------------------------- #
def _choose_edge_maps(maps: list[dict], use_unique: bool = True) -> list[tuple[int, ...]]:
    perms = [tuple(m["permutation"]) for m in maps]
    if use_unique:
        perms = sorted(set(perms))
    if not perms:
        return []
    # Keep largest label cardinality if mixed sizes are present.  Holonomy only
    # makes sense among maps on a common fiber S.
    counts = Counter(len(p) for p in perms)
    n = max(counts, key=lambda k: (counts[k], k))
    return [p for p in perms if len(p) == n]


def _iter_loops(perms: list[tuple[int, ...]], loop_length: int, max_loops: int, rng: np.random.Generator):
    if not perms:
        return
    if max_loops and max_loops > 0:
        for _ in range(int(max_loops)):
            yield tuple(perms[int(rng.integers(0, len(perms)))] for _ in range(int(loop_length)))
    else:
        # Exhaustive over the supplied map set.  This is intended for unique
        # generators; e.g. 4^4=256 for the observed S3 sector.
        yield from itertools.product(perms, repeat=int(loop_length))


def gauge_covariance_trial(edges: tuple[tuple[int, ...], ...], group: list[tuple[int, ...]], rng: np.random.Generator) -> dict:
    """Random local relabeling test for one loop."""
    if not edges:
        return {"covariant": False, "same_conjugacy_type": False}
    n = len(edges[0])
    if not group:
        group = [identity(n)]
    L = len(edges)
    # Local relabelings g_0,...,g_{L-1}; endpoint equals g_0.
    gs = [group[int(rng.integers(0, len(group)))] for _ in range(L)]
    h = holonomy_product(edges)
    transformed_edges: list[tuple[int, ...]] = []
    for i, e in enumerate(edges):
        g_left = gs[i]
        g_right = gs[(i + 1) % L]
        e_prime = compose(g_right, compose(e, inverse(g_left)))
        transformed_edges.append(e_prime)
    h_prime = holonomy_product(transformed_edges)
    expected = conjugate(gs[0], h)
    return dict(
        covariant=bool(h_prime == expected),
        same_conjugacy_type=bool(cycle_lengths(h_prime) == cycle_lengths(h)),
        holonomy=tuple(h),
        transformed_holonomy=tuple(h_prime),
        expected=tuple(expected),
    )


def run_loop_holonomy(
    winner_path: str,
    top_n: int = 50,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    min_transport: float = 0.80,
    min_source: float = 0.80,
    min_accuracy: float = 0.95,
    min_residual: float = 1e-9,
    loop_length: int = 4,
    use_unique: bool = True,
    max_loops: int = 0,
    gauge_trials_per_loop: int = 3,
    base_seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(int(base_seed) % 2**32)
    maps, algebra_summary = extract_transport_permutations(
        winner_path,
        top_n=int(top_n),
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        source_for_liveness=source_for_liveness,
        min_transport=float(min_transport),
        min_source=float(min_source),
        min_accuracy=float(min_accuracy),
        min_residual=float(min_residual),
        base_seed=int(base_seed),
    )
    perms = _choose_edge_maps(maps, use_unique=bool(use_unique))
    if not perms:
        summary = dict(
            verdict="NO LIVE BIJECTIVE TRANSPORT MAPS FOR LOOP HOLONOMY",
            n_transport_maps=int(len(maps)),
            n_edge_maps=0,
            algebra_summary=algebra_summary,
        )
        return pd.DataFrame(), summary
    n = len(perms[0])
    group = sorted(A.generate_group(perms, n=n), key=lambda p: (perm_type(p), p))
    group_summary = A.classify_group(set(group), n=n)

    rows: list[dict] = []
    for li, edges in enumerate(_iter_loops(perms, int(loop_length), int(max_loops), rng)):
        h = holonomy_product(edges)
        cov_ok = 0
        class_ok = 0
        trials = max(0, int(gauge_trials_per_loop))
        for _ in range(trials):
            t = gauge_covariance_trial(tuple(edges), group, rng)
            cov_ok += int(t["covariant"])
            class_ok += int(t["same_conjugacy_type"])
        h_type = perm_type(h)
        rows.append(
            dict(
                loop_index=int(li),
                n=int(n),
                loop_length=int(loop_length),
                holonomy_json=json.dumps(list(h)),
                holonomy_type=h_type,
                holonomy_order=int(perm_order(h)),
                nontrivial=bool(h != identity(n)),
                edge_types_json=json.dumps([perm_type(e) for e in edges]),
                edges_json=json.dumps([list(e) for e in edges]),
                gauge_trials=int(trials),
                gauge_covariance_success=float(cov_ok / trials) if trials else math.nan,
                conjugacy_class_success=float(class_ok / trials) if trials else math.nan,
            )
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        summary = dict(verdict="NO LOOPS ENUMERATED", n_transport_maps=int(len(maps)), n_edge_maps=int(len(perms)))
        return df, summary
    type_counts = Counter(df.holonomy_type.astype(str))
    unique_h = {tuple(json.loads(x)) for x in df.holonomy_json.astype(str)}
    nontriv = float(df.nontrivial.mean())
    cov_success = float(np.nanmean(df.gauge_covariance_success.values)) if "gauge_covariance_success" in df else math.nan
    class_success = float(np.nanmean(df.conjugacy_class_success.values)) if "conjugacy_class_success" in df else math.nan
    if nontriv > 0.0 and cov_success > 0.999 and class_success > 0.999:
        verdict = f"NONTRIVIAL FINITE HOLONOMY: {group_summary.get('group_name')} loop sector has gauge-covariant non-identity classes"
    elif nontriv > 0.0:
        verdict = "NONTRIVIAL LOOP PRODUCTS FOUND, BUT GAUGE-COVARIANCE TEST NEEDS REVIEW"
    else:
        verdict = "TRIVIAL HOLONOMY: transported maps compose to identity in tested loops"
    summary = dict(
        verdict=verdict,
        n_transport_maps=int(len(maps)),
        n_edge_maps=int(len(perms)),
        unique_edge_maps=[list(p) for p in perms],
        generated_group=group_summary,
        group_elements=[list(p) for p in group],
        loop_count=int(len(df)),
        holonomy_type_counts={str(k): int(v) for k, v in type_counts.items()},
        unique_holonomy_count=int(len(unique_h)),
        group_coverage_fraction=float(len(unique_h) / max(len(group), 1)),
        nontrivial_holonomy_fraction=nontriv,
        gauge_covariance_success=cov_success,
        conjugacy_class_success=class_success,
        algebra_summary=algebra_summary,
        filters=dict(min_transport=float(min_transport), min_source=float(min_source), min_accuracy=float(min_accuracy), min_residual=float(min_residual)),
    )
    return df, summary


# --------------------------------------------------------------------------- #
# Plot / CLI helpers
# --------------------------------------------------------------------------- #
def plot_loop_holonomy(df: pd.DataFrame, summary: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None or len(df) == 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    counts = Counter(df.holonomy_type.astype(str))
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    ax[0].bar(range(len(vals)), vals)
    ax[0].set_xticks(range(len(vals)))
    ax[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax[0].set_ylabel("count")
    ax[0].set_title("loop holonomy conjugacy/cycle type")

    # Holonomy order histogram.
    oc = Counter(df.holonomy_order.astype(int))
    olabels = list(sorted(oc.keys()))
    ax[1].bar(range(len(olabels)), [oc[k] for k in olabels])
    ax[1].set_xticks(range(len(olabels)))
    ax[1].set_xticklabels([str(k) for k in olabels])
    ax[1].set_ylabel("count")
    ax[1].set_title("holonomy order")

    # Gauge covariance success.
    ax[2].hist(df.gauge_covariance_success.dropna().values, bins=np.linspace(0, 1, 11))
    ax[2].set_xlabel("covariance success")
    ax[2].set_ylabel("loops")
    ax[2].set_title("local relabeling covariance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _jsonify(x):
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if math.isnan(float(x)):
            return None
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Loop/holonomy audit for transported temporal-chain labels.")
    p.add_argument("winner_pickle")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--initial-mode", choices=["source_all", "source_random", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--source-for-liveness", choices=["initial", "postB"], default="postB")
    p.add_argument("--min-transport", type=float, default=0.80)
    p.add_argument("--min-source", type=float, default=0.80)
    p.add_argument("--min-accuracy", type=float, default=0.95)
    p.add_argument("--min-residual", type=float, default=1e-9)
    p.add_argument("--loop-length", type=int, default=4)
    p.add_argument("--use-all-maps", action="store_true", help="Use all selected maps, not just unique permutations.")
    p.add_argument("--max-loops", type=int, default=0, help="0 means exhaustive over the chosen map set; otherwise sample this many loops.")
    p.add_argument("--gauge-trials-per-loop", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", default="example_results/loop_holonomy.csv")
    p.add_argument("--plot", default="")
    args = p.parse_args(argv)
    df, summary = run_loop_holonomy(
        args.winner_pickle,
        top_n=int(args.top_n),
        initial_mode=args.initial_mode,  # type: ignore[arg-type]
        n_random_initial=int(args.n_random_initial),
        source_for_liveness=args.source_for_liveness,  # type: ignore[arg-type]
        min_transport=float(args.min_transport),
        min_source=float(args.min_source),
        min_accuracy=float(args.min_accuracy),
        min_residual=float(args.min_residual),
        loop_length=int(args.loop_length),
        use_unique=not bool(args.use_all_maps),
        max_loops=int(args.max_loops),
        gauge_trials_per_loop=int(args.gauge_trials_per_loop),
        base_seed=int(args.base_seed),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)
    if args.plot:
        plot_loop_holonomy(df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(_jsonify(summary), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
