"""
transportalgebra.py -- algebra audit for transported temporal-chain labels.

Stage-1 gauge-structure audit
-----------------------------
Single-diamond equalizers can be formal/source-dead.  The temporal-chain
control and seeded-recovery tests identify the more meaningful object:
transported labels S1 -> S2 across two sequential diamond convergences.

This module loads saved TemporalChain winners (for example from
``seededtemporalrecovery.py --save-winners``), reconstructs the exact labels at
both convergence points, and extracts the induced empirical map

    phi : S1 -> S2.

For high-transport winners phi should be nearly deterministic.  When |S1|=|S2|
and phi is bijective, we classify its permutation cycle type.  The set of such
permutations, across winners, generates a finite subgroup of Sym(|S|).  This is
the algebra you need before designing loop/holonomy experiments.

Important caution
-----------------
A single edge map S1 -> S2 is not by itself gauge-invariant, because the two
local alphabets may be relabeled independently.  Cycle types/groups reported
here use the canonical labels produced by the equalizer algorithm and should be
read as a first structural audit.  True gauge-invariant content appears in loop
holonomies, where local relabelings conjugate the loop product.

Recommended use
---------------
python -m relgauge.seededtemporalrecovery 4 ^
  --ks-B 2 --ks-M 2 --ks-A 2 --ws 1 ^
  --seed-ensembles copy,permutive,block,canalizing,random ^
  --seed-mutation-rates 0.05,0.1 ^
  --runs 5 --population 20 --generations 60 ^
  --fitness live_sum --target interface --proposal-mode mixed ^
  --entry-rate 0.04 --table-rate 0.03 ^
  --initial-mode joint_random --n-random-initial 4096 ^
  --save-winners example_results/seeded_temporal_winners_q4.pkl ^
  --save-winner-top-n 50 ^
  --out example_results/seeded_temporal_recovery_q4_saved.csv

python -m relgauge.transportalgebra example_results/seeded_temporal_winners_q4.pkl ^
  --top-n 50 --initial-mode joint_random --n-random-initial 4096 ^
  --min-transport 0.8 --min-source 0.8 --min-accuracy 0.95 ^
  --out example_results/transport_algebra_q4.csv ^
  --plot example_results/fig_transport_algebra_q4.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter, defaultdict, deque
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import temporalchain as T

InitialMode = T.InitialMode


# --------------------------------------------------------------------------- #
# Loading winners
# --------------------------------------------------------------------------- #
def load_temporal_winners(path: str) -> list[dict]:
    """Load saved TemporalChain winners from seeded/blind temporal modules.

    Returns dictionaries with at least ``candidate`` and usually ``row``/``score``.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "winners" in obj:
        winners = list(obj["winners"])
    elif isinstance(obj, list):
        winners = list(obj)
    else:
        raise ValueError("unrecognized winner pickle; expected dict with 'winners' or a list")
    out: list[dict] = []
    for i, item in enumerate(winners):
        if isinstance(item, dict):
            cand = item.get("candidate") or item.get("tc") or item.get("system")
            if cand is None:
                continue
            row = item.get("row", {}) if isinstance(item.get("row", {}), dict) else {}
            out.append({"winner_index": i, "candidate": cand, "row": dict(row), "score": float(item.get("score", row.get("recovery_fitness", math.nan)))})
        elif isinstance(item, tuple) and len(item) >= 3:
            score, row, cand = item[0], item[1], item[2]
            out.append({"winner_index": i, "candidate": cand, "row": dict(row) if isinstance(row, dict) else {}, "score": float(score)})
        else:
            # Maybe a raw TemporalChain.
            if hasattr(item, "joint") and hasattr(item, "offA1"):
                out.append({"winner_index": i, "candidate": item, "row": {}, "score": math.nan})
    return out


# --------------------------------------------------------------------------- #
# Information helpers
# --------------------------------------------------------------------------- #
def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _entropy_discrete(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    _, counts = np.unique(np.asarray(x, dtype=np.int64), return_counts=True)
    return _entropy_from_counts(counts)


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if x.shape != y.shape or len(x) == 0:
        return 0.0
    xv, xi = np.unique(x, return_inverse=True)
    yv, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xv), len(yv)), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)
    return float(
        _entropy_from_counts(joint.sum(axis=1))
        + _entropy_from_counts(joint.sum(axis=0))
        - _entropy_from_counts(joint.reshape(-1))
    )


def _best_map(x: np.ndarray, y: np.ndarray) -> tuple[dict[int, int], float, np.ndarray]:
    """Best deterministic y=f(x), returned as map, accuracy, predictions."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if x.shape != y.shape or len(x) == 0:
        return {}, 0.0, np.zeros(0, dtype=np.int64)
    mapping: dict[int, int] = {}
    pred = np.empty_like(y)
    correct = 0
    for a in np.unique(x):
        mask = x == a
        ys = y[mask]
        vals, counts = np.unique(ys, return_counts=True)
        b = int(vals[int(np.argmax(counts))])
        mapping[int(a)] = b
        pred[mask] = b
        correct += int(counts.max())
    return mapping, float(correct / len(x)), pred


# --------------------------------------------------------------------------- #
# Extract labels and maps
# --------------------------------------------------------------------------- #
def temporal_chain_labels(
    tc: T.TemporalChain,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    rng: np.random.Generator | None = None,
) -> dict:
    """Return source, S1, S2, and exact-equalizer metadata for a chain."""
    states0 = T.initial_states(tc, mode=initial_mode, n_random=int(n_random_initial), rng=rng)
    sched = T.canonical_schedules(tc)
    states_postB = T._step_subset(tc.joint, states0, sched["postB"])
    states_mid = T._step_subset(tc.joint, states0, sched["mid"])
    states_final = T._step_subset(tc.joint, states0, sched["final"])

    source_initial = T._word(states0, tc.q, tc.b_boundary)
    source_postB = T._word(states_postB, tc.q, tc.b_boundary)
    source = source_initial if source_for_liveness == "initial" else source_postB

    a1_left = T._word(states_mid, tc.q, tc.a1_left)
    a1_right = T._word(states_mid, tc.q, tc.a1_right)
    a2_left = T._word(states_final, tc.q, tc.a2_left)
    a2_right = T._word(states_final, tc.q, tc.a2_right)
    eq1 = T.exact_equalizer_labels(a1_left, a1_right, tc.q, tc.w)
    eq2 = T.exact_equalizer_labels(a2_left, a2_right, tc.q, tc.w)
    return dict(
        source=source,
        s1=np.asarray(eq1["labels"], dtype=np.int64),
        s2=np.asarray(eq2["labels"], dtype=np.int64),
        eq1=eq1,
        eq2=eq2,
        states0=states0,
    )


def induced_transport_map(source: np.ndarray, s1: np.ndarray, s2: np.ndarray, q: int, w: int) -> dict:
    s1 = np.asarray(s1, dtype=np.int64)
    s2 = np.asarray(s2, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    h1 = _entropy_discrete(s1)
    h2 = _entropy_discrete(s2)
    hs = _entropy_discrete(source)
    mi12 = _mutual_info_discrete(s1, s2)
    mis2 = _mutual_info_discrete(source, s2)
    mis1 = _mutual_info_discrete(source, s1)
    mapping, acc, pred = _best_map(s1, s2)

    u1 = sorted(int(x) for x in np.unique(s1))
    u2 = sorted(int(x) for x in np.unique(s2))
    relabel1 = {a: i for i, a in enumerate(u1)}
    relabel2 = {b: i for i, b in enumerate(u2)}
    n1 = len(u1)
    n2 = len(u2)
    perm: tuple[int, ...] | None = None
    is_bijection = False
    if n1 == n2 and n1 > 0 and acc >= 1.0 - 1e-12 and set(mapping.keys()) == set(u1) and set(mapping.values()) == set(u2):
        perm = tuple(int(relabel2[mapping[a]]) for a in u1)
        is_bijection = sorted(perm) == list(range(n1))
    return dict(
        H_s1_bits=float(h1),
        H_s2_bits=float(h2),
        H_source_bits=float(hs),
        transport_mi_bits=float(mi12),
        transport_mi_over_Hs1=float(mi12 / h1) if h1 > 1e-12 else 0.0,
        transport_mi_over_Hs2=float(mi12 / h2) if h2 > 1e-12 else 0.0,
        source_to_s1_mi_bits=float(mis1),
        source_to_s2_mi_bits=float(mis2),
        source_to_s1_mi_over_Hs1=float(mis1 / h1) if h1 > 1e-12 else 0.0,
        source_to_s2_mi_over_Hs2=float(mis2 / h2) if h2 > 1e-12 else 0.0,
        best_accuracy=float(acc),
        n_s1_classes=int(n1),
        n_s2_classes=int(n2),
        mapping={int(k): int(v) for k, v in mapping.items()},
        is_bijection=bool(is_bijection),
        permutation=perm,
    )


# --------------------------------------------------------------------------- #
# Permutation/group classification
# --------------------------------------------------------------------------- #
def _compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """p after q: i -> p[q[i]]."""
    return tuple(int(p[int(q[i])]) for i in range(len(p)))


def _identity(n: int) -> tuple[int, ...]:
    return tuple(range(int(n)))


def _cycle_lengths(p: tuple[int, ...]) -> tuple[int, ...]:
    n = len(p)
    seen = [False] * n
    out: list[int] = []
    for i in range(n):
        if seen[i]:
            continue
        cur = i
        l = 0
        while not seen[cur]:
            seen[cur] = True
            l += 1
            cur = int(p[cur])
        out.append(l)
    return tuple(sorted(out, reverse=True))


def _perm_order(p: tuple[int, ...]) -> int:
    lcm = 1
    for l in _cycle_lengths(p):
        lcm = abs(lcm * int(l)) // math.gcd(lcm, int(l))
    return int(lcm)


def _parity(p: tuple[int, ...]) -> int:
    inv = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            inv += int(p[i] > p[j])
    return inv % 2


def generate_group(gens: Iterable[tuple[int, ...]], n: int, max_size: int = 100000) -> set[tuple[int, ...]]:
    gens = [tuple(int(x) for x in g) for g in gens if len(g) == int(n)]
    if not gens:
        return {_identity(n)}
    ident = _identity(n)
    group: set[tuple[int, ...]] = {ident}
    queue = deque([ident])
    gen_set = list({g for g in gens} | {ident})
    while queue:
        a = queue.popleft()
        for g in gen_set:
            for h in (_compose(a, g), _compose(g, a)):
                if h not in group:
                    group.add(h)
                    queue.append(h)
                    if len(group) > int(max_size):
                        return group
    return group


def classify_group(group: set[tuple[int, ...]], n: int) -> dict:
    order = len(group)
    orders = Counter(_perm_order(g) for g in group)
    cycle_types = Counter(str(_cycle_lengths(g)) for g in group)
    all_even = all(_parity(g) == 0 for g in group)
    abelian = True
    glist = list(group)
    for i, a in enumerate(glist):
        for b in glist[i + 1:]:
            if _compose(a, b) != _compose(b, a):
                abelian = False
                break
        if not abelian:
            break
    cyclic = any(_perm_order(g) == order for g in group)
    # Transitive action on {0,...,n-1}
    orbit = set()
    for g in group:
        orbit.add(int(g[0]))
    transitive = len(orbit) == int(n)
    name = f"order_{order}"
    if order == 1:
        name = "trivial"
    elif order == 2 and cyclic:
        name = "C2"
    elif order == 3 and cyclic:
        name = "C3"
    elif order == 4:
        name = "C4" if cyclic else "V4"
    elif n == 4 and order == 8:
        name = "D4_or_order8"
    elif n == 4 and order == 12 and all_even:
        name = "A4"
    elif n == 4 and order == 24:
        name = "S4"
    elif cyclic:
        name = f"C{order}"
    return dict(
        group_name=name,
        group_order=int(order),
        group_cyclic=bool(cyclic),
        group_abelian=bool(abelian),
        group_transitive=bool(transitive),
        group_all_even=bool(all_even),
        element_order_hist={str(k): int(v) for k, v in sorted(orders.items())},
        cycle_type_hist={str(k): int(v) for k, v in sorted(cycle_types.items())},
    )


def classify_permutation(p: tuple[int, ...] | None) -> str:
    if p is None:
        return "nonbijective"
    ct = _cycle_lengths(tuple(p))
    if all(x == 1 for x in ct):
        return "identity"
    return "cycle_" + "_".join(str(x) for x in ct)


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #
def audit_transport_algebra(
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
) -> tuple[pd.DataFrame, dict]:
    winners = load_temporal_winners(winner_path)[: int(top_n)]
    rows: list[dict] = []
    perms_by_n: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for item in winners:
        idx = int(item["winner_index"])
        tc = item["candidate"]
        rng = np.random.default_rng((int(base_seed) * 65537 + idx * 1009) % 2**32)
        try:
            labs = temporal_chain_labels(
                tc,
                initial_mode=initial_mode,
                n_random_initial=int(n_random_initial),
                source_for_liveness=source_for_liveness,
                rng=rng,
            )
            m = induced_transport_map(labs["source"], labs["s1"], labs["s2"], tc.q, tc.w)
            eq1 = labs["eq1"]
            eq2 = labs["eq2"]
            selected = bool(
                float(eq1["residual_qcoords"]) >= float(min_residual)
                and float(eq2["residual_qcoords"]) >= float(min_residual)
                and float(m["transport_mi_over_Hs2"]) >= float(min_transport)
                and float(m["source_to_s2_mi_over_Hs2"]) >= float(min_source)
                and float(m["best_accuracy"]) >= float(min_accuracy)
            )
            perm = m["permutation"]
            if selected and m["is_bijection"] and perm is not None:
                perms_by_n[len(perm)].append(tuple(perm))
            row = dict(
                winner_index=idx,
                score=float(item.get("score", math.nan)),
                seed_ensemble=str(item.get("row", {}).get("seed_ensemble", getattr(tc, "meta", {}).get("seed_ensemble", "unknown"))),
                seed_mutation_rate=float(item.get("row", {}).get("seed_mutation_rate", getattr(tc, "meta", {}).get("seed_mutation_rate", math.nan))),
                q=int(tc.q),
                w=int(tc.w),
                s1_shared_classes=int(eq1["shared_classes"]),
                s2_shared_classes=int(eq2["shared_classes"]),
                s1_residual_qcoords=float(eq1["residual_qcoords"]),
                s2_residual_qcoords=float(eq2["residual_qcoords"]),
                selected_for_group=bool(selected),
                permutation_type=classify_permutation(perm),
                permutation_json=json.dumps(list(perm) if perm is not None else None),
                mapping_json=json.dumps(m["mapping"]),
                **{k: v for k, v in m.items() if k not in {"mapping", "permutation"}},
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            row = dict(winner_index=idx, error=str(exc), selected_for_group=False, permutation_type="error")
        rows.append(row)
    df = pd.DataFrame(rows)
    group_summaries = {}
    for n, perms in sorted(perms_by_n.items()):
        group = generate_group(perms, n=n)
        group_summaries[str(n)] = classify_group(group, n=n)
        group_summaries[str(n)]["n_generators"] = int(len(perms))
        group_summaries[str(n)]["unique_generators"] = int(len(set(perms)))
        group_summaries[str(n)]["generators"] = [list(p) for p in sorted(set(perms))[:50]]
    if len(group_summaries):
        # Use largest-n group as primary.
        primary_n = sorted(int(k) for k in group_summaries.keys())[-1]
        primary = group_summaries[str(primary_n)]
        if primary.get("group_name", "") in {"C4", "C3", "C2"}:
            verdict = f"CYCLIC TRANSPORT ALGEBRA: transported labels generate {primary['group_name']} on |S|={primary_n}"
        elif primary.get("group_name") in {"S4", "A4", "D4_or_order8", "V4"}:
            verdict = f"NONTRIVIAL TRANSPORT ALGEBRA: transported labels generate {primary['group_name']} on |S|={primary_n}"
        else:
            verdict = f"TRANSPORT ALGEBRA FOUND: group {primary.get('group_name')} on |S|={primary_n}"
    else:
        verdict = "NO BIJECTIVE LIVE-TRANSPORT MAPS PASSED FILTERS"
    summary = dict(
        verdict=verdict,
        n_rows=int(len(df)),
        n_selected=int(df.get("selected_for_group", pd.Series(dtype=bool)).sum()) if len(df) else 0,
        permutation_type_counts={str(k): int(v) for k, v in Counter(df.get("permutation_type", pd.Series(dtype=str)).dropna().astype(str)).items()},
        group_summaries=group_summaries,
        filters=dict(min_transport=float(min_transport), min_source=float(min_source), min_accuracy=float(min_accuracy), min_residual=float(min_residual)),
    )
    return df, summary


def plot_transport_algebra(df: pd.DataFrame, summary: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None or len(df) == 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # Transport vs source scatter.
    ax[0].scatter(df.get("source_to_s2_mi_over_Hs2", 0), df.get("transport_mi_over_Hs2", 0), s=20, alpha=0.7)
    ax[0].set_xlabel(r"$I(source;S_2)/H(S_2)$")
    ax[0].set_ylabel(r"$I(S_1;S_2)/H(S_2)$")
    ax[0].set_title("liveness vs transport")
    # Permutation types.
    counts = Counter(df["permutation_type"].astype(str))
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    ax[1].bar(range(len(vals)), vals)
    ax[1].set_xticks(range(len(vals)))
    ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax[1].set_title("induced map type")
    ax[1].set_ylabel("count")
    # Group orders.
    gs = summary.get("group_summaries", {})
    if gs:
        g_labels = [f"|S|={k}\n{v.get('group_name')}" for k, v in gs.items()]
        g_vals = [v.get("group_order", 0) for v in gs.values()]
        ax[2].bar(range(len(g_vals)), g_vals)
        ax[2].set_xticks(range(len(g_vals)))
        ax[2].set_xticklabels(g_labels, rotation=0, fontsize=8)
        ax[2].set_ylabel("generated group order")
        ax[2].set_title("generated transport group")
    else:
        ax[2].text(0.5, 0.5, "no group passed filters", ha="center", va="center")
        ax[2].set_axis_off()
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
    p = argparse.ArgumentParser(description="Audit the algebra of transported labels S1 -> S2 in saved temporal-chain winners.")
    p.add_argument("winner_pickle")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--initial-mode", choices=["source_all", "source_random", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--source-for-liveness", choices=["initial", "postB"], default="postB")
    p.add_argument("--min-transport", type=float, default=0.80)
    p.add_argument("--min-source", type=float, default=0.80)
    p.add_argument("--min-accuracy", type=float, default=0.95)
    p.add_argument("--min-residual", type=float, default=1e-9)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", default="example_results/transport_algebra.csv")
    p.add_argument("--plot", default="")
    args = p.parse_args(argv)
    df, summary = audit_transport_algebra(
        args.winner_pickle,
        top_n=int(args.top_n),
        initial_mode=args.initial_mode,  # type: ignore[arg-type]
        n_random_initial=int(args.n_random_initial),
        source_for_liveness=args.source_for_liveness,  # type: ignore[arg-type]
        min_transport=float(args.min_transport),
        min_source=float(args.min_source),
        min_accuracy=float(args.min_accuracy),
        min_residual=float(args.min_residual),
        base_seed=int(args.base_seed),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)
    if args.plot:
        plot_transport_algebra(df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(_jsonify(summary), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
