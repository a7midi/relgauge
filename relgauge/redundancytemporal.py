"""
redundancytemporal.py -- capacity-vs-redundancy test for transported labels.

Motivation
----------
The w=2 seeded temporal-chain runs showed that wider interfaces can carry richer
live labels, but do not automatically enlarge the live-transport basin.  This
module separates two uses of width:

  capacity_*   : each boundary coordinate may carry an independent source symbol
                 (higher information rate, more alignment constraints).
  repeat_*     : all boundary coordinates redundantly carry the same source label
                 (lower information rate, intended mutation/error robustness).

The question is whether increasing width can be used as redundancy rather than
mere capacity.  A positive redundancy signal is:

  repeat_* has higher final live fraction / mutation survival than capacity_*
  at the same q,w,mutation rate, while preserving nonzero live transported S.

This is still a finite experiment over exact shared alphabets and live temporal
transport; it does not assume a continuum, geometry, or gauge group.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import temporalchain as T
from . import blindtemporalchain as B
from . import seededtemporalrecovery as S

RedundancySeed = Literal[
    "random",
    "capacity_copy", "capacity_permutive", "capacity_block", "capacity_canalizing",
    "repeat_copy", "repeat_permutive", "repeat_block", "repeat_canalizing",
]


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_strings(text: str) -> tuple[str, ...]:
    return tuple(str(x.strip()) for x in str(text).split(",") if str(x).strip())


def _stable_seed(*items: object) -> int:
    s = 0
    for item in items:
        txt = str(item)
        for i, ch in enumerate(txt):
            s = (s * 1315423911 + (i + 1) * ord(ch) + 0x9E3779B9) & 0xFFFFFFFF
    return int(s)


def _mutate_targets(tc: T.TemporalChain, rng: np.random.Generator, rate: float, mutate_all_rules: bool) -> None:
    if float(rate) <= 0:
        return
    targets = range(tc.k_total) if mutate_all_rules else T._interface_targets(tc)  # type: ignore[attr-defined]
    for v in targets:
        tc.joint.rules[int(v)] = T._mutate_rule(tc.joint.rules[int(v)], tc.q, float(rate), rng)  # type: ignore[attr-defined]


def _ensure_pred(tc: T.TemporalChain, target: int, source: int) -> None:
    """Add source as predecessor of target if the base topology lacks that edge.

    Repeat-coded width intentionally fans one source label into all boundary
    wires, whereas the default temporal-chain topology only connects B[j] to
    the j-th branch wire.  When we need B[0] -> branch[j], add the edge and
    resize the rule table; the rule is immediately overwritten by _set_copy.
    """
    target = int(target); source = int(source)
    preds = tuple(int(x) for x in tc.joint.preds[target])
    if source in preds:
        return
    new_preds = tuple(sorted(set(preds) | {source}))
    tc.joint.preds[target] = new_preds
    tc.joint.rules[target] = np.zeros(tc.q ** len(new_preds), dtype=np.int64)


def _set_repeat_transport(
    tc: T.TemporalChain,
    rng: np.random.Generator,
    mode: Literal["copy", "permutive", "block", "canalizing"],
    n_blocks: int | None = None,
) -> dict:
    """Configure a temporal chain to redundantly transport ONE source label over w wires.

    The repeated label is B[0] after B's own update.  Every first-diamond branch
    receives the same transformed source label; the second diamond then copies
    the selected panel label forward.  Thus width is used for redundancy, not for
    w independent source coordinates.
    """
    T._force_B_label_availability(tc)  # type: ignore[attr-defined]
    q = int(tc.q)
    b0 = int(tc.offB)  # first B boundary vertex is the repeated source label.
    if mode == "copy":
        t_first = None
        meta = {"mode": "repeat_copy"}
    elif mode == "permutive":
        # Use the SAME permutation on both branches and all redundant wires.
        # This preserves a q-ary label up to relabeling while keeping left/right aligned.
        p = rng.permutation(q).astype(np.int64)
        t_first = T._perm_transform(p)  # type: ignore[attr-defined]
        meta = {"mode": "repeat_permutive", "perm": p.tolist()}
    elif mode == "block":
        t_first = T._block_transform(q, n_blocks)  # type: ignore[attr-defined]
        meta = {"mode": "repeat_block", "n_blocks": n_blocks}
    elif mode == "canalizing":
        t_first = T._canalizing_transform(q)  # type: ignore[attr-defined]
        meta = {"mode": "repeat_canalizing"}
    else:  # pragma: no cover
        raise ValueError(mode)

    for j in range(int(tc.w)):
        # First diamond: same B label is repeated to all C1/D1 boundary wires.
        c1 = int(tc.offC1 + j)
        d1 = int(tc.offD1 + j)
        _ensure_pred(tc, c1, b0)
        _ensure_pred(tc, d1, b0)
        T._set_copy(tc, c1, b0, t_first)  # type: ignore[attr-defined]
        T._set_copy(tc, d1, b0, t_first)  # type: ignore[attr-defined]

        # C1/D1 panels are copied into A1.
        a1l = int(tc.offA1 + j)
        a1r = int(tc.offA1 + tc.w + j)
        T._set_copy(tc, a1l, c1, None)  # type: ignore[attr-defined]
        T._set_copy(tc, a1r, d1, None)  # type: ignore[attr-defined]

        # Second diamond: copy the repeated A1 panel labels forward.
        c2 = int(tc.offC2 + j)
        d2 = int(tc.offD2 + j)
        T._set_copy(tc, c2, a1l, None)  # type: ignore[attr-defined]
        T._set_copy(tc, d2, a1r, None)  # type: ignore[attr-defined]

        a2l = int(tc.offA2 + j)
        a2r = int(tc.offA2 + tc.w + j)
        T._set_copy(tc, a2l, c2, None)  # type: ignore[attr-defined]
        T._set_copy(tc, a2r, d2, None)  # type: ignore[attr-defined]
    return meta


def make_redundancy_candidate(
    kB: int,
    kM: int,
    kA: int,
    q: int,
    w: int,
    rng: np.random.Generator,
    seed_ensemble: RedundancySeed = "repeat_copy",
    seed_mutation_rate: float = 0.05,
    seed_mutate_all_rules: bool = True,
    n_blocks: int | None = None,
) -> T.TemporalChain:
    """Build either a capacity-coded or redundancy-coded temporal transport seed."""
    ens = str(seed_ensemble)
    if ens == "random":
        tc = T.make_rule_temporal_chain(kB, kM, kA, q, w, rng, rule_ensemble="random", mutation_rate=0.0)
    elif ens.startswith("capacity_"):
        base = ens.replace("capacity_", "")
        if base not in {"copy", "permutive", "block", "canalizing"}:
            raise ValueError(f"unknown capacity seed {seed_ensemble!r}")
        # Existing temporalchain ensemble: width can carry independent coordinates.
        tc = T.make_rule_temporal_chain(
            kB, kM, kA, q, w, rng,
            rule_ensemble=base,  # type: ignore[arg-type]
            mutation_rate=float(seed_mutation_rate),
            n_blocks=n_blocks,
            mutate_all_rules=bool(seed_mutate_all_rules),
        )
        tc.meta.update(coding_mode="capacity", seed_base=base)
    elif ens.startswith("repeat_"):
        base = ens.replace("repeat_", "")
        if base not in {"copy", "permutive", "block", "canalizing"}:
            raise ValueError(f"unknown repeat seed {seed_ensemble!r}")
        tc = T.make_temporal_chain(kB, kM, kA, q, w, rng)
        meta = _set_repeat_transport(tc, rng, base, n_blocks=n_blocks)  # type: ignore[arg-type]
        _mutate_targets(tc, rng, float(seed_mutation_rate), bool(seed_mutate_all_rules))
        tc.meta.update(coding_mode="repeat", seed_base=base, transforms=meta)
    else:
        raise ValueError(f"unknown seed ensemble {seed_ensemble!r}")

    tc.meta.update(
        rule_ensemble="redundancy_seeded",
        seed_ensemble=str(seed_ensemble),
        seed_mutation_rate=float(seed_mutation_rate),
        seed_mutate_all_rules=bool(seed_mutate_all_rules),
    )
    tc.joint.meta.update(tc.meta)
    return tc


def evaluate_candidate(
    tc: T.TemporalChain,
    initial_mode: T.InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    rng: np.random.Generator | None = None,
    fitness: S.FitnessMode = "live_sum",
    transport_threshold: float = 0.80,
    source_threshold: float = 0.80,
    accuracy_threshold: float = 0.95,
) -> dict:
    m = S.evaluate_recovery_candidate(
        tc,
        initial_mode=initial_mode,
        n_random_initial=n_random_initial,
        source_for_liveness=source_for_liveness,
        rng=rng,
        fitness=fitness,
        transport_threshold=transport_threshold,
        source_threshold=source_threshold,
        accuracy_threshold=accuracy_threshold,
    )
    # Yield measures distinguish "many low-capacity live channels" from
    # "few high-capacity live channels".
    m["live_capacity_yield"] = float(m["min_residual_qcoords"] if m["selected_recovered_live_transport"] else 0.0)
    m["coding_mode"] = str(tc.meta.get("coding_mode", "random"))
    m["seed_base"] = str(tc.meta.get("seed_base", "random"))
    return m


def _rank_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        float(row.get("recovery_fitness", 0.0)),
        float(row.get("source_to_s2_mi_over_Hs2", 0.0)),
        float(row.get("transport_mi_over_Hs2", 0.0)),
        float(row.get("min_residual_qcoords", 0.0)),
        float(row.get("s1_to_s2_best_accuracy", 0.0)),
    )


def run_one_recovery(
    kB: int,
    kM: int,
    kA: int,
    w: int,
    q: int,
    seed_ensemble: RedundancySeed,
    seed_mutation_rate: float,
    runs: int = 5,
    population: int = 20,
    generations: int = 60,
    fitness: S.FitnessMode = "live_sum",
    target: B.TargetMode = "interface",
    proposal_mode: B.ProposalMode = "mixed",
    entry_rate: float = 0.04,
    table_rate: float = 0.03,
    n_table_mutations: int = 1,
    elite_fraction: float = 0.25,
    reseed_fraction: float = 0.0,
    seed_mutate_all_rules: bool = True,
    initial_mode: T.InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    transport_threshold: float = 0.80,
    source_threshold: float = 0.80,
    accuracy_threshold: float = 0.95,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[list[dict], list[tuple[float, dict, T.TemporalChain]]]:
    rows: list[dict] = []
    saved: list[tuple[float, dict, T.TemporalChain]] = []
    for run in range(int(runs)):
        rng = np.random.default_rng(_stable_seed("redundancy", base_seed, kB, kM, kA, w, q, seed_ensemble, seed_mutation_rate, run) % 2**32)
        pop = [
            make_redundancy_candidate(kB, kM, kA, q, w, rng, seed_ensemble, seed_mutation_rate, seed_mutate_all_rules)
            for _ in range(int(population))
        ]
        for gen in range(int(generations) + 1):
            evals: list[tuple[float, dict, T.TemporalChain]] = []
            for ci, cand in enumerate(pop):
                erng = np.random.default_rng(_stable_seed("eval", base_seed, seed_ensemble, seed_mutation_rate, run, gen, ci) % 2**32)
                m = evaluate_candidate(
                    cand,
                    initial_mode=initial_mode,
                    n_random_initial=n_random_initial,
                    source_for_liveness=source_for_liveness,
                    rng=erng,
                    fitness=fitness,
                    transport_threshold=transport_threshold,
                    source_threshold=source_threshold,
                    accuracy_threshold=accuracy_threshold,
                )
                m.update(search_mode="redundancy_recovery", run=int(run), generation=int(gen), candidate=int(ci), target_mode=str(target), proposal_mode=str(proposal_mode))
                rows.append(m)
                evals.append((float(m["recovery_fitness"]), m, cand))
            evals.sort(key=lambda t: _rank_key(t[1]), reverse=True)
            best = evals[0][1]
            saved.append((float(best["recovery_fitness"]), dict(best), B.clone_temporal_chain(evals[0][2])))
            if verbose:
                print(
                    "redundancy-recovery "
                    f"ens={seed_ensemble} mu={seed_mutation_rate:g} run={run} gen={gen} "
                    f"best={best['recovery_fitness']:.4f} minres={best['min_residual_qcoords']:.4f} "
                    f"trans={best['transport_mi_over_Hs2']:.4f} src={best['source_to_s2_mi_over_Hs2']:.4f} "
                    f"live={best['selected_recovered_live_transport']}",
                    flush=True,
                )
            if gen >= int(generations):
                break
            n_elite = max(1, int(math.ceil(float(elite_fraction) * int(population))))
            elites = [B.clone_temporal_chain(c) for _score, _m, c in evals[:n_elite]]
            new_pop: list[T.TemporalChain] = [B.clone_temporal_chain(e) for e in elites]
            while len(new_pop) < int(population):
                if reseed_fraction > 0 and rng.random() < float(reseed_fraction):
                    new_pop.append(make_redundancy_candidate(kB, kM, kA, q, w, rng, seed_ensemble, seed_mutation_rate, seed_mutate_all_rules))
                    continue
                parent = elites[int(rng.integers(0, len(elites)))]
                child, _info = B.mutate_temporal_chain(
                    parent,
                    rng,
                    target=target,
                    proposal_mode=proposal_mode,
                    entry_rate=float(entry_rate),
                    table_rate=float(table_rate),
                    n_table_mutations=int(n_table_mutations),
                )
                child.meta.update(seed_ensemble=str(seed_ensemble), seed_mutation_rate=float(seed_mutation_rate), coding_mode=str(parent.meta.get("coding_mode", "unknown")), seed_base=str(parent.meta.get("seed_base", "unknown")))
                child.joint.meta.update(child.meta)
                new_pop.append(child)
            pop = new_pop
    return rows, saved


def run_redundancy_temporal_recovery(
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (4,),
    ws: Iterable[int] = (2,),
    q: int = 4,
    seed_ensembles: Iterable[RedundancySeed] = ("capacity_copy", "repeat_copy", "capacity_permutive", "repeat_permutive", "capacity_block", "repeat_block", "random"),
    seed_mutation_rates: Iterable[float] = (0.05, 0.10),
    runs: int = 5,
    population: int = 20,
    generations: int = 60,
    fitness: S.FitnessMode = "live_sum",
    target: B.TargetMode = "interface",
    proposal_mode: B.ProposalMode = "mixed",
    entry_rate: float = 0.04,
    table_rate: float = 0.03,
    n_table_mutations: int = 1,
    elite_fraction: float = 0.25,
    reseed_fraction: float = 0.0,
    seed_mutate_all_rules: bool = True,
    initial_mode: T.InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    transport_threshold: float = 0.80,
    source_threshold: float = 0.80,
    accuracy_threshold: float = 0.95,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[float, dict, T.TemporalChain]]]:
    all_rows: list[dict] = []
    all_saved: list[tuple[float, dict, T.TemporalChain]] = []
    for kB in ks_B:
        for kM in ks_M:
            for kA in ks_A:
                for w in ws:
                    if w > min(kB, kM) or kA < 2 * w:
                        continue
                    for ens in seed_ensembles:
                        for mu in seed_mutation_rates:
                            rows, saved = run_one_recovery(
                                int(kB), int(kM), int(kA), int(w), int(q), ens, float(mu),
                                runs=runs,
                                population=population,
                                generations=generations,
                                fitness=fitness,
                                target=target,
                                proposal_mode=proposal_mode,
                                entry_rate=entry_rate,
                                table_rate=table_rate,
                                n_table_mutations=n_table_mutations,
                                elite_fraction=elite_fraction,
                                reseed_fraction=reseed_fraction,
                                seed_mutate_all_rules=seed_mutate_all_rules,
                                initial_mode=initial_mode,
                                n_random_initial=n_random_initial,
                                source_for_liveness=source_for_liveness,
                                transport_threshold=transport_threshold,
                                source_threshold=source_threshold,
                                accuracy_threshold=accuracy_threshold,
                                base_seed=base_seed,
                                verbose=verbose,
                            )
                            all_rows.extend(rows)
                            all_saved.extend(saved)
                            if verbose:
                                print(f"redundancy-recovery complete ens={ens}, mu={mu:g}, kB={kB}, kM={kM}, kA={kA}, w={w}", flush=True)
    return pd.DataFrame(all_rows), all_saved


def analyze_redundancy_recovery(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"verdict": "NO DATA"}
    rows = []
    for keys, g in df.groupby(["seed_ensemble", "seed_mutation_rate", "kB", "kM", "kA", "w"], dropna=False):
        ens, mu, kB, kM, kA, w = keys
        gen0 = g[g.generation == int(g.generation.min())]
        genf = g[g.generation == int(g.generation.max())]
        best0 = gen0.sort_values("recovery_fitness", ascending=False).head(1).iloc[0]
        bestf = genf.sort_values("recovery_fitness", ascending=False).head(1).iloc[0]
        final_live_fraction = float(genf.selected_recovered_live_transport.mean())
        final_best_min_residual = float(bestf.min_residual_qcoords)
        rows.append({
            "seed_ensemble": str(ens),
            "coding_mode": str(bestf.get("coding_mode", "unknown")),
            "seed_base": str(bestf.get("seed_base", "unknown")),
            "seed_mutation_rate": float(mu),
            "kB": int(kB), "kM": int(kM), "kA": int(kA), "w": int(w),
            "initial_live_fraction": float(gen0.selected_recovered_live_transport.mean()),
            "final_live_fraction": final_live_fraction,
            "recovery_gain": float(final_live_fraction - gen0.selected_recovered_live_transport.mean()),
            "initial_best_min_residual": float(best0.min_residual_qcoords),
            "final_best_min_residual": final_best_min_residual,
            "final_best_transport": float(bestf.transport_mi_over_Hs2),
            "final_best_source_to_s2": float(bestf.source_to_s2_mi_over_Hs2),
            "final_best_fitness": float(bestf.recovery_fitness),
            "final_best_live": bool(bestf.selected_recovered_live_transport),
            "live_capacity_yield": float(final_live_fraction * final_best_min_residual),
        })
    by_df = pd.DataFrame(rows)
    comparisons = []
    for keys, sub in by_df.groupby(["seed_base", "seed_mutation_rate", "kB", "kM", "kA", "w"], dropna=False):
        base, mu, kB, kM, kA, w = keys
        cap = sub[sub.coding_mode == "capacity"]
        rep = sub[sub.coding_mode == "repeat"]
        if len(cap) and len(rep):
            c = cap.iloc[0]
            r = rep.iloc[0]
            comparisons.append({
                "seed_base": str(base),
                "seed_mutation_rate": float(mu),
                "kB": int(kB), "kM": int(kM), "kA": int(kA), "w": int(w),
                "capacity_final_live_fraction": float(c.final_live_fraction),
                "repeat_final_live_fraction": float(r.final_live_fraction),
                "repeat_minus_capacity_live_fraction": float(r.final_live_fraction - c.final_live_fraction),
                "capacity_final_residual": float(c.final_best_min_residual),
                "repeat_final_residual": float(r.final_best_min_residual),
                "repeat_minus_capacity_residual": float(r.final_best_min_residual - c.final_best_min_residual),
                "capacity_live_yield": float(c.live_capacity_yield),
                "repeat_live_yield": float(r.live_capacity_yield),
                "repeat_minus_capacity_yield": float(r.live_capacity_yield - c.live_capacity_yield),
            })
    comp_df = pd.DataFrame(comparisons)
    if len(comp_df):
        mean_live_delta = float(comp_df.repeat_minus_capacity_live_fraction.mean())
        mean_res_delta = float(comp_df.repeat_minus_capacity_residual.mean())
        mean_yield_delta = float(comp_df.repeat_minus_capacity_yield.mean())
    else:
        mean_live_delta = mean_res_delta = mean_yield_delta = math.nan

    random = by_df[by_df.seed_ensemble == "random"]
    random_live = float(random.final_live_fraction.mean()) if len(random) else 0.0
    if not math.isnan(mean_live_delta) and mean_live_delta > 0.10 and random_live < 0.05:
        verdict = "REDUNDANCY SIGNAL: repeated labels enlarge the live-transport basin relative to capacity-coded width"
    elif not math.isnan(mean_yield_delta) and mean_yield_delta > 0.05 and random_live < 0.05:
        verdict = "REDUNDANCY-YIELD SIGNAL: repeated labels improve live capacity yield, though live fraction alone is mixed"
    elif not math.isnan(mean_res_delta) and mean_res_delta < -0.10 and random_live < 0.05:
        verdict = "CAPACITY/REDUNDANCY TRADEOFF: capacity-coded width carries richer labels; redundancy does not dominate basin size"
    else:
        verdict = "NO CLEAR REDUNDANCY ADVANTAGE in this regime"
    return {
        "verdict": verdict,
        "n_rows": int(len(df)),
        "random_final_live_fraction": random_live,
        "mean_repeat_minus_capacity_live_fraction": mean_live_delta,
        "mean_repeat_minus_capacity_residual": mean_res_delta,
        "mean_repeat_minus_capacity_yield": mean_yield_delta,
        "by_seed": rows,
        "capacity_vs_repeat": comparisons,
    }


def plot_redundancy_recovery(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    # final generation summaries for plotting
    maxgen = df.groupby(["seed_ensemble", "seed_mutation_rate"])["generation"].transform("max")
    final = df[df.generation == maxgen].copy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for ens in sorted(final.seed_ensemble.unique()):
        sub = final[final.seed_ensemble == ens]
        g = sub.groupby("seed_mutation_rate")["selected_recovered_live_transport"].mean().dropna()
        ax[0].plot(g.index, g.values, "o-", label=str(ens))
    ax[0].set_xlabel("seed mutation rate")
    ax[0].set_ylabel("final live fraction")
    ax[0].set_title("basin size / mutation survival")
    ax[0].legend(fontsize=7)

    # best residual by final generation
    for ens in sorted(final.seed_ensemble.unique()):
        vals = []
        xs = []
        for mu, sub in final[final.seed_ensemble == ens].groupby("seed_mutation_rate"):
            vals.append(float(sub.sort_values("recovery_fitness", ascending=False).iloc[0].min_residual_qcoords))
            xs.append(float(mu))
        ax[1].plot(xs, vals, "o-", label=str(ens))
    ax[1].set_xlabel("seed mutation rate")
    ax[1].set_ylabel("best min residual")
    ax[1].set_title("transported label capacity")

    for ens in sorted(final.seed_ensemble.unique()):
        vals = []
        xs = []
        for mu, sub in final[final.seed_ensemble == ens].groupby("seed_mutation_rate"):
            live = float(sub.selected_recovered_live_transport.mean())
            bestres = float(sub.sort_values("recovery_fitness", ascending=False).iloc[0].min_residual_qcoords)
            vals.append(live * bestres)
            xs.append(float(mu))
        ax[2].plot(xs, vals, "o-", label=str(ens))
    ax[2].set_xlabel("seed mutation rate")
    ax[2].set_ylabel("live fraction × best residual")
    ax[2].set_title("live capacity yield")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, tuple):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if math.isnan(x) else x
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj


def save_winners(path: str, saved: list[tuple[float, dict, T.TemporalChain]], top_n: int = 50) -> None:
    saved_sorted = sorted(saved, key=lambda t: _rank_key(t[1]), reverse=True)[: int(top_n)]
    payload = []
    for score, metrics, tc in saved_sorted:
        payload.append({"score": float(score), "metrics": dict(metrics), "candidate": tc})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"kind": "redundancy_temporal_winners", "winners": payload}, f)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("q", type=int, nargs="?", default=4)
    ap.add_argument("--ks-B", default="2")
    ap.add_argument("--ks-M", default="2")
    ap.add_argument("--ks-A", default="4")
    ap.add_argument("--ws", default="2")
    ap.add_argument("--seed-ensembles", default="capacity_copy,repeat_copy,capacity_permutive,repeat_permutive,capacity_block,repeat_block,random")
    ap.add_argument("--seed-mutation-rates", default="0.02,0.05,0.1")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--fitness", choices=("live_sum", "live_product", "transport_sum"), default="live_sum")
    ap.add_argument("--target", choices=("all", "interface"), default="interface")
    ap.add_argument("--proposal-mode", choices=("entry", "table", "mixed"), default="mixed")
    ap.add_argument("--entry-rate", type=float, default=0.04)
    ap.add_argument("--table-rate", type=float, default=0.03)
    ap.add_argument("--n-table-mutations", type=int, default=1)
    ap.add_argument("--elite-fraction", type=float, default=0.25)
    ap.add_argument("--reseed-fraction", type=float, default=0.0)
    ap.add_argument("--seed-mutate-interface-only", action="store_true")
    ap.add_argument("--initial-mode", choices=("source_all", "source_random", "joint_random"), default="joint_random")
    ap.add_argument("--n-random-initial", type=int, default=4096)
    ap.add_argument("--source-for-liveness", choices=("initial", "postB"), default="postB")
    ap.add_argument("--transport-threshold", type=float, default=0.80)
    ap.add_argument("--source-threshold", type=float, default=0.80)
    ap.add_argument("--accuracy-threshold", type=float, default=0.95)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", default="example_results/redundancy_temporal_recovery.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--save-winners", default=None)
    ap.add_argument("--save-winner-top-n", type=int, default=50)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df, saved = run_redundancy_temporal_recovery(
        ks_B=_parse_ints(args.ks_B),
        ks_M=_parse_ints(args.ks_M),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        q=int(args.q),
        seed_ensembles=_parse_strings(args.seed_ensembles),  # type: ignore[arg-type]
        seed_mutation_rates=_parse_floats(args.seed_mutation_rates),
        runs=int(args.runs),
        population=int(args.population),
        generations=int(args.generations),
        fitness=args.fitness,  # type: ignore[arg-type]
        target=args.target,  # type: ignore[arg-type]
        proposal_mode=args.proposal_mode,  # type: ignore[arg-type]
        entry_rate=float(args.entry_rate),
        table_rate=float(args.table_rate),
        n_table_mutations=int(args.n_table_mutations),
        elite_fraction=float(args.elite_fraction),
        reseed_fraction=float(args.reseed_fraction),
        seed_mutate_all_rules=not bool(args.seed_mutate_interface_only),
        initial_mode=args.initial_mode,  # type: ignore[arg-type]
        n_random_initial=int(args.n_random_initial),
        source_for_liveness=args.source_for_liveness,  # type: ignore[arg-type]
        transport_threshold=float(args.transport_threshold),
        source_threshold=float(args.source_threshold),
        accuracy_threshold=float(args.accuracy_threshold),
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = analyze_redundancy_recovery(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)
    if args.plot:
        plot_redundancy_recovery(df, args.plot)
        print(f"wrote {args.plot}")
    if args.save_winners:
        save_winners(args.save_winners, saved, top_n=int(args.save_winner_top_n))
        print(f"wrote {args.save_winners}")
    print(json.dumps(_jsonify(summary), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
