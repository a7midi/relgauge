"""
squareholonomy.py -- path-dependence / plaquette holonomy for transported labels.

This is the step after transported temporal-chain labels have been found.
A single diamond is a convergence constraint; it does not by itself define
curvature.  Here each EDGE of a square is a temporal chain carrying a live label:

    A --AB--> B --BD--> D
    |                 ^
    AC                CD
    v                 |
    C ---------------+

For each edge e we construct one temporal-chain system and extract its induced
transport map

    phi_e : S_in -> S_out.

The square holonomy / path-dependence observable is

    Delta = (phi_BD o phi_AB)^(-1) o (phi_CD o phi_AC).

If Delta is non-identity, the two paths A->B->D and A->C->D differ. Under local
relabelings g_A,g_B,g_C,g_D of the transported label basis, Delta transforms by
conjugation at A, so its conjugacy class is the gauge-invariant finite curvature
observable.

Important scope note
--------------------
This module builds a *single square instance* as four temporal-chain edge
modules generated together from one seed/ensemble.  The edge modules are
separable at the microscopic graph level, so this is a first finite plaquette
model of path dependence for transported labels.  It is stricter than merely
multiplying maps sampled from unrelated saved winners, and it is the right test
before building a fully shared-corner microscopic network.

Recommended first run
---------------------
python -m relgauge.squareholonomy 4 ^
  --ks-B 2 --ks-M 2 --ks-A 2 --ws 1 ^
  --instances 50 ^
  --edge-ensembles copy,permutive,block,canalizing,random ^
  --mutation-rates 0,0.02,0.05 ^
  --initial-mode joint_random --n-random-initial 4096 ^
  --out example_results/square_holonomy_q4_w1.csv ^
  --plot example_results/fig_square_holonomy_q4_w1.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import temporalchain as T
from . import transportalgebra as A
from . import loopholonomy as H

InitialMode = T.InitialMode
RuleEnsemble = T.RuleEnsemble

EDGE_NAMES = ("AB", "BD", "AC", "CD")


@dataclass
class EdgeTransport:
    name: str
    chain: T.TemporalChain
    selected: bool
    permutation: tuple[int, ...] | None
    permutation_type: str
    n: int
    transport: float
    source: float
    accuracy: float
    s1_residual: float
    s2_residual: float
    seed_ensemble: str
    mutation_rate: float


def _edge_transport(
    name: str,
    tc: T.TemporalChain,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    min_transport: float = 0.80,
    min_source: float = 0.80,
    min_accuracy: float = 0.95,
    min_residual: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> EdgeTransport:
    labs = A.temporal_chain_labels(
        tc,
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        source_for_liveness=source_for_liveness,
        rng=rng,
    )
    m = A.induced_transport_map(labs["source"], labs["s1"], labs["s2"], tc.q, tc.w)
    eq1 = labs["eq1"]
    eq2 = labs["eq2"]
    perm = m.get("permutation")
    if perm is not None:
        perm = tuple(int(x) for x in perm)
    selected = bool(
        float(eq1["residual_qcoords"]) >= float(min_residual)
        and float(eq2["residual_qcoords"]) >= float(min_residual)
        and float(m["transport_mi_over_Hs2"]) >= float(min_transport)
        and float(m["source_to_s2_mi_over_Hs2"]) >= float(min_source)
        and float(m["best_accuracy"]) >= float(min_accuracy)
        and bool(m["is_bijection"])
        and perm is not None
    )
    return EdgeTransport(
        name=str(name),
        chain=tc,
        selected=selected,
        permutation=perm,
        permutation_type=A.classify_permutation(perm),
        n=int(len(perm)) if perm is not None else 0,
        transport=float(m.get("transport_mi_over_Hs2", 0.0)),
        source=float(m.get("source_to_s2_mi_over_Hs2", 0.0)),
        accuracy=float(m.get("best_accuracy", 0.0)),
        s1_residual=float(eq1["residual_qcoords"]),
        s2_residual=float(eq2["residual_qcoords"]),
        seed_ensemble=str(tc.meta.get("rule_ensemble", "unknown")),
        mutation_rate=float(tc.meta.get("mutation_rate", math.nan)),
    )


def make_square_edges(
    kB: int,
    kM: int,
    kA: int,
    q: int,
    w: int,
    rng: np.random.Generator,
    edge_ensemble: RuleEnsemble = "permutive",
    mutation_rate: float = 0.0,
    n_blocks: int | None = None,
    mutate_all_rules: bool = True,
) -> dict[str, T.TemporalChain]:
    """Generate four temporal-chain edge modules as one square instance."""
    out: dict[str, T.TemporalChain] = {}
    for name in EDGE_NAMES:
        tc = T.make_rule_temporal_chain(
            kB=int(kB),
            kM=int(kM),
            kA=int(kA),
            q=int(q),
            w=int(w),
            rng=rng,
            rule_ensemble=edge_ensemble,
            mutation_rate=float(mutation_rate),
            n_blocks=n_blocks,
            mutate_all_rules=bool(mutate_all_rules),
        )
        tc.meta.update(square_edge=name, square_ensemble=str(edge_ensemble), square_mutation_rate=float(mutation_rate))
        out[name] = tc
    return out


def _all_same_cardinality(edges: list[EdgeTransport]) -> bool:
    ns = {e.n for e in edges if e.selected and e.permutation is not None}
    return len(ns) == 1 and bool(ns)


def _path_delta(ab: tuple[int, ...], bd: tuple[int, ...], ac: tuple[int, ...], cd: tuple[int, ...]) -> dict:
    """Return top, bottom and Delta=(top)^-1 bottom."""
    top = H.compose(bd, ab)
    bottom = H.compose(cd, ac)
    delta = H.compose(H.inverse(top), bottom)
    return dict(top=top, bottom=bottom, delta=delta)


def _gauge_covariance_square(
    ab: tuple[int, ...],
    bd: tuple[int, ...],
    ac: tuple[int, ...],
    cd: tuple[int, ...],
    group: list[tuple[int, ...]],
    rng: np.random.Generator,
) -> dict:
    """Local relabeling covariance for Delta=(BD AB)^-1 (CD AC).

    Edge transforms:
      AB -> g_B AB g_A^{-1}
      BD -> g_D BD g_B^{-1}
      AC -> g_C AC g_A^{-1}
      CD -> g_D CD g_C^{-1}
    Therefore Delta -> g_A Delta g_A^{-1}.
    """
    n = len(ab)
    if not group:
        group = [H.identity(n)]
    gA = group[int(rng.integers(0, len(group)))]
    gB = group[int(rng.integers(0, len(group)))]
    gC = group[int(rng.integers(0, len(group)))]
    gD = group[int(rng.integers(0, len(group)))]
    raw = _path_delta(ab, bd, ac, cd)
    abp = H.compose(gB, H.compose(ab, H.inverse(gA)))
    bdp = H.compose(gD, H.compose(bd, H.inverse(gB)))
    acp = H.compose(gC, H.compose(ac, H.inverse(gA)))
    cdp = H.compose(gD, H.compose(cd, H.inverse(gC)))
    trans = _path_delta(abp, bdp, acp, cdp)
    expected = H.conjugate(gA, raw["delta"])
    return dict(
        covariant=bool(trans["delta"] == expected),
        same_conjugacy_type=bool(H.cycle_lengths(trans["delta"]) == H.cycle_lengths(raw["delta"])),
        delta=raw["delta"],
        transformed_delta=trans["delta"],
        expected=expected,
    )


def measure_square_instance(
    edges: dict[str, T.TemporalChain],
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    min_transport: float = 0.80,
    min_source: float = 0.80,
    min_accuracy: float = 0.95,
    min_residual: float = 1e-9,
    gauge_trials: int = 10,
    base_seed: int = 0,
) -> dict:
    rng = np.random.default_rng(int(base_seed) % 2**32)
    et: dict[str, EdgeTransport] = {}
    for name in EDGE_NAMES:
        et[name] = _edge_transport(
            name,
            edges[name],
            initial_mode=initial_mode,
            n_random_initial=int(n_random_initial),
            source_for_liveness=source_for_liveness,
            min_transport=float(min_transport),
            min_source=float(min_source),
            min_accuracy=float(min_accuracy),
            min_residual=float(min_residual),
            rng=rng,
        )
    edge_list = [et[k] for k in EDGE_NAMES]
    all_selected = all(e.selected for e in edge_list)
    same_n = _all_same_cardinality(edge_list)
    row: dict = {
        "all_edges_selected": bool(all_selected),
        "same_label_cardinality": bool(same_n),
        "edge_selected_json": json.dumps({k: bool(et[k].selected) for k in EDGE_NAMES}),
        "edge_types_json": json.dumps({k: et[k].permutation_type for k in EDGE_NAMES}),
        "edge_perms_json": json.dumps({k: list(et[k].permutation) if et[k].permutation is not None else None for k in EDGE_NAMES}),
        "min_edge_transport": float(min(e.transport for e in edge_list)),
        "min_edge_source": float(min(e.source for e in edge_list)),
        "min_edge_accuracy": float(min(e.accuracy for e in edge_list)),
        "min_edge_residual": float(min(min(e.s1_residual, e.s2_residual) for e in edge_list)),
        "label_cardinality": int(edge_list[0].n) if same_n else 0,
    }
    for k in EDGE_NAMES:
        e = et[k]
        row.update({
            f"{k}_selected": bool(e.selected),
            f"{k}_type": e.permutation_type,
            f"{k}_transport": float(e.transport),
            f"{k}_source": float(e.source),
            f"{k}_accuracy": float(e.accuracy),
            f"{k}_s1_residual": float(e.s1_residual),
            f"{k}_s2_residual": float(e.s2_residual),
        })
    if not (all_selected and same_n):
        row.update(
            path_holonomy_valid=False,
            delta_json=json.dumps(None),
            delta_type="invalid",
            delta_order=0,
            nontrivial_path_holonomy=False,
            gauge_covariance_success=math.nan,
            conjugacy_class_success=math.nan,
            generated_group_name="invalid",
            generated_group_order=0,
        )
        return row
    ab, bd, ac, cd = (et["AB"].permutation, et["BD"].permutation, et["AC"].permutation, et["CD"].permutation)
    assert ab is not None and bd is not None and ac is not None and cd is not None
    n = len(ab)
    perms = [ab, bd, ac, cd]
    group = sorted(A.generate_group(perms, n=n), key=lambda p: (A.classify_permutation(p), p))
    gsum = A.classify_group(set(group), n=n)
    hp = _path_delta(ab, bd, ac, cd)
    delta = hp["delta"]
    cov = 0
    cls = 0
    for _ in range(max(0, int(gauge_trials))):
        t = _gauge_covariance_square(ab, bd, ac, cd, group, rng)
        cov += int(t["covariant"])
        cls += int(t["same_conjugacy_type"])
    row.update(
        path_holonomy_valid=True,
        top_json=json.dumps(list(hp["top"])),
        bottom_json=json.dumps(list(hp["bottom"])),
        delta_json=json.dumps(list(delta)),
        delta_type=A.classify_permutation(delta),
        delta_order=int(H.perm_order(delta)),
        nontrivial_path_holonomy=bool(delta != H.identity(n)),
        gauge_trials=int(gauge_trials),
        gauge_covariance_success=float(cov / gauge_trials) if gauge_trials else math.nan,
        conjugacy_class_success=float(cls / gauge_trials) if gauge_trials else math.nan,
        generated_group_name=str(gsum.get("group_name", "unknown")),
        generated_group_order=int(gsum.get("group_order", 0)),
        generated_group_abelian=bool(gsum.get("group_abelian", False)),
        generated_group_cyclic=bool(gsum.get("group_cyclic", False)),
        generated_group_transitive=bool(gsum.get("group_transitive", False)),
        group_elements_json=json.dumps([list(p) for p in group]),
    )
    return row


def run_square_holonomy_sweep(
    q: int = 4,
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2,),
    ws: Iterable[int] = (1,),
    edge_ensembles: Iterable[RuleEnsemble] = ("copy", "permutive", "block", "canalizing", "random"),
    mutation_rates: Iterable[float] = (0.0,),
    instances: int = 50,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    min_transport: float = 0.80,
    min_source: float = 0.80,
    min_accuracy: float = 0.95,
    min_residual: float = 1e-9,
    gauge_trials: int = 10,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for kB in ks_B:
        for kM in ks_M:
            for kA in ks_A:
                for w in ws:
                    if int(w) > min(int(kB), int(kM)) or int(kA) < 2 * int(w):
                        if verbose:
                            print(f"skip kB={kB}, kM={kM}, kA={kA}, w={w}: invalid dimensions", flush=True)
                        continue
                    for ens in edge_ensembles:
                        for mu in mutation_rates:
                            for inst in range(int(instances)):
                                seed = (int(base_seed) * 1000003 + int(q) * 10007 + int(kB) * 1009 + int(kM) * 101 + int(kA) * 17 + int(w) * 31 + hash(str(ens)) % 100000 + int(round(float(mu) * 10000)) * 19 + inst) % 2**32
                                rng = np.random.default_rng(seed)
                                edges = make_square_edges(
                                    kB=int(kB), kM=int(kM), kA=int(kA), q=int(q), w=int(w),
                                    rng=rng, edge_ensemble=ens, mutation_rate=float(mu),
                                )
                                row = measure_square_instance(
                                    edges,
                                    initial_mode=initial_mode,
                                    n_random_initial=int(n_random_initial),
                                    source_for_liveness=source_for_liveness,
                                    min_transport=float(min_transport),
                                    min_source=float(min_source),
                                    min_accuracy=float(min_accuracy),
                                    min_residual=float(min_residual),
                                    gauge_trials=int(gauge_trials),
                                    base_seed=seed + 12345,
                                )
                                row.update(q=int(q), kB=int(kB), kM=int(kM), kA=int(kA), w=int(w), edge_ensemble=str(ens), mutation_rate=float(mu), instance=int(inst), seed=int(seed))
                                rows.append(row)
                            if verbose:
                                print(f"square-holonomy ensemble={ens}, mu={mu}, kB={kB}, kM={kM}, kA={kA}, w={w} done", flush=True)
    df = pd.DataFrame(rows)
    summary = analyze_square_holonomy(df)
    return df, summary


def analyze_square_holonomy(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"verdict": "NO DATA"}
    valid = df[df["path_holonomy_valid"] == True] if "path_holonomy_valid" in df else pd.DataFrame()
    by = []
    for (ens, mu), g in df.groupby(["edge_ensemble", "mutation_rate"], dropna=False):
        gv = g[g["path_holonomy_valid"] == True]
        by.append({
            "edge_ensemble": str(ens),
            "mutation_rate": float(mu),
            "n": int(len(g)),
            "valid_fraction": float(len(gv) / len(g)) if len(g) else 0.0,
            "nontrivial_path_fraction": float(gv["nontrivial_path_holonomy"].mean()) if len(gv) else 0.0,
            "mean_min_edge_transport": float(g["min_edge_transport"].mean()) if "min_edge_transport" in g else math.nan,
            "mean_min_edge_source": float(g["min_edge_source"].mean()) if "min_edge_source" in g else math.nan,
            "mean_min_edge_residual": float(g["min_edge_residual"].mean()) if "min_edge_residual" in g else math.nan,
            "delta_type_counts": {str(k): int(v) for k, v in Counter(gv["delta_type"].astype(str)).items()} if len(gv) else {},
            "group_name_counts": {str(k): int(v) for k, v in Counter(gv["generated_group_name"].astype(str)).items()} if len(gv) else {},
            "gauge_covariance_success": float(np.nanmean(gv["gauge_covariance_success"])) if len(gv) else math.nan,
            "conjugacy_class_success": float(np.nanmean(gv["conjugacy_class_success"])) if len(gv) else math.nan,
        })
    valid_fraction = float(len(valid) / len(df)) if len(df) else 0.0
    nontriv = float(valid["nontrivial_path_holonomy"].mean()) if len(valid) else 0.0
    cov = float(np.nanmean(valid["gauge_covariance_success"])) if len(valid) else math.nan
    if valid_fraction > 0 and nontriv > 0.05 and (math.isnan(cov) or cov > 0.999):
        verdict = "NONTRIVIAL PATH HOLONOMY: single-square edge systems show gauge-covariant path dependence"
    elif valid_fraction > 0 and nontriv == 0:
        verdict = "FLAT SQUARE CONNECTION: transported edge maps exist but tested plaquettes are path-independent"
    else:
        verdict = "NO VALID SQUARE HOLONOMY in this regime"
    return {
        "verdict": verdict,
        "n_rows": int(len(df)),
        "valid_square_fraction": valid_fraction,
        "nontrivial_path_holonomy_fraction": nontriv,
        "gauge_covariance_success": cov,
        "conjugacy_class_success": float(np.nanmean(valid["conjugacy_class_success"])) if len(valid) else math.nan,
        "delta_type_counts": {str(k): int(v) for k, v in Counter(valid["delta_type"].astype(str)).items()} if len(valid) else {},
        "generated_group_counts": {str(k): int(v) for k, v in Counter(valid["generated_group_name"].astype(str)).items()} if len(valid) else {},
        "by_ensemble": by,
        "scope_note": "Each row is one square instance with four temporal-chain edge modules generated together from one seed/ensemble. Edge modules are separable; this is a plaquette-level transport test before a fully shared-corner microscopic network.",
    }


def plot_square_holonomy(df: pd.DataFrame, summary: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None or len(df) == 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # Nontrivial path fraction by ensemble/mutation.
    for ens, g in df.groupby("edge_ensemble"):
        xs = []
        ys = []
        for mu, h in sorted(g.groupby("mutation_rate"), key=lambda kv: kv[0]):
            hv = h[h["path_holonomy_valid"] == True]
            xs.append(float(mu))
            ys.append(float(hv["nontrivial_path_holonomy"].mean()) if len(hv) else 0.0)
        ax[0].plot(xs, ys, "o-", label=str(ens))
    ax[0].set_xlabel("mutation rate")
    ax[0].set_ylabel("nontrivial path fraction")
    ax[0].set_title("path-dependence in one square")
    ax[0].legend(fontsize=8)
    # Valid square fraction.
    for ens, g in df.groupby("edge_ensemble"):
        xs = []
        ys = []
        for mu, h in sorted(g.groupby("mutation_rate"), key=lambda kv: kv[0]):
            xs.append(float(mu))
            ys.append(float(h["path_holonomy_valid"].mean()))
        ax[1].plot(xs, ys, "o-", label=str(ens))
    ax[1].set_xlabel("mutation rate")
    ax[1].set_ylabel("valid square fraction")
    ax[1].set_title("all four edges live/bijective")
    # Holonomy types.
    valid = df[df["path_holonomy_valid"] == True]
    counts = Counter(valid["delta_type"].astype(str)) if len(valid) else Counter()
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    ax[2].bar(range(len(vals)), vals)
    ax[2].set_xticks(range(len(vals)))
    ax[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax[2].set_ylabel("count")
    ax[2].set_title("path holonomy type")
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


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def _parse_strs(s: str) -> list[str]:
    return [str(x).strip() for x in str(s).split(",") if str(x).strip()]


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Square path-dependence / plaquette holonomy for transported temporal-chain labels.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--ks-B", default="2")
    p.add_argument("--ks-M", default="2")
    p.add_argument("--ks-A", default="2")
    p.add_argument("--ws", default="1")
    p.add_argument("--instances", type=int, default=50)
    p.add_argument("--edge-ensembles", default="copy,permutive,block,canalizing,random")
    p.add_argument("--mutation-rates", default="0")
    p.add_argument("--initial-mode", choices=["source_all", "source_random", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--source-for-liveness", choices=["initial", "postB"], default="postB")
    p.add_argument("--min-transport", type=float, default=0.80)
    p.add_argument("--min-source", type=float, default=0.80)
    p.add_argument("--min-accuracy", type=float, default=0.95)
    p.add_argument("--min-residual", type=float, default=1e-9)
    p.add_argument("--gauge-trials", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", default="example_results/square_holonomy.csv")
    p.add_argument("--plot", default="")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    df, summary = run_square_holonomy_sweep(
        q=int(args.q),
        ks_B=_parse_ints(args.ks_B),
        ks_M=_parse_ints(args.ks_M),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        edge_ensembles=_parse_strs(args.edge_ensembles),
        mutation_rates=_parse_floats(args.mutation_rates),
        instances=int(args.instances),
        initial_mode=args.initial_mode,  # type: ignore[arg-type]
        n_random_initial=int(args.n_random_initial),
        source_for_liveness=args.source_for_liveness,  # type: ignore[arg-type]
        min_transport=float(args.min_transport),
        min_source=float(args.min_source),
        min_accuracy=float(args.min_accuracy),
        min_residual=float(args.min_residual),
        gauge_trials=int(args.gauge_trials),
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)
    if args.plot:
        plot_square_holonomy(df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(_jsonify(summary), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
