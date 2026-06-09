"""
seededtemporalrecovery.py -- can consistency selection repair noisy live transport?

Why this exists
---------------
``temporalchain.py`` showed that structured chains (copy, permutive, block,
canalizing) transport live labels through two convergences, while random chains
do not.  ``blindtemporalchain.py`` showed that starting from fully random rule
space, the present blind mutation/search routine does not discover live
transport.

This module tests the missing middle case:

    Start inside or near a known transport basin, corrupt the rules by mutation,
    then let a blind consistency/transport objective evolve the rule tables.

If the search restores high live-transport metrics from noisy copy/permutive
seeds, the fitness is useful inside the transport basin and the random failure is
a search-distance/alignment problem.  If recovery fails even from noisy seeds,
the fitness or topology needs redesign.

Recommended run
---------------
python -m relgauge.seededtemporalrecovery 4 ^
  --ks-B 2 --ks-M 2 --ks-A 2 --ws 1 ^
  --seed-ensembles copy,permutive,block,canalizing,random ^
  --seed-mutation-rates 0.05,0.1 ^
  --runs 5 --population 20 --generations 60 ^
  --fitness live_sum ^
  --target interface --proposal-mode mixed ^
  --entry-rate 0.04 --table-rate 0.03 ^
  --initial-mode joint_random --n-random-initial 4096 ^
  --out example_results/seeded_temporal_recovery_q4.csv ^
  --plot example_results/fig_seeded_temporal_recovery_q4.png
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

SeedEnsemble = Literal["random", "copy", "permutive", "block", "canalizing", "constant"]
FitnessMode = Literal["live_sum", "live_product", "transport_sum"]


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


def make_seeded_candidate(
    kB: int,
    kM: int,
    kA: int,
    q: int,
    w: int,
    rng: np.random.Generator,
    seed_ensemble: SeedEnsemble = "copy",
    seed_mutation_rate: float = 0.05,
    seed_mutate_all_rules: bool = True,
    n_blocks: int | None = None,
) -> T.TemporalChain:
    """Construct a temporal chain from a known ensemble, then corrupt it.

    This uses the same structured ensembles and mutation semantics as
    ``temporalchain.py``.  The search objective never sees the seed ensemble; it
    is only used to initialize near a known transport basin.
    """
    tc = T.make_rule_temporal_chain(
        kB=int(kB),
        kM=int(kM),
        kA=int(kA),
        q=int(q),
        w=int(w),
        rng=rng,
        rule_ensemble=seed_ensemble,  # type: ignore[arg-type]
        mutation_rate=float(seed_mutation_rate),
        n_blocks=n_blocks,
        mutate_all_rules=bool(seed_mutate_all_rules),
    )
    tc.joint.meta["rule_ensemble"] = "seeded"
    tc.joint.meta["seed_ensemble"] = str(seed_ensemble)
    tc.joint.meta["seed_mutation_rate"] = float(seed_mutation_rate)
    tc.meta.update(
        rule_ensemble="seeded",
        seed_ensemble=str(seed_ensemble),
        seed_mutation_rate=float(seed_mutation_rate),
        seed_mutate_all_rules=bool(seed_mutate_all_rules),
    )
    return tc


def evaluate_recovery_candidate(
    tc: T.TemporalChain,
    initial_mode: T.InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    rng: np.random.Generator | None = None,
    fitness: FitnessMode = "live_sum",
    transport_threshold: float = 0.80,
    source_threshold: float = 0.80,
    accuracy_threshold: float = 0.95,
) -> dict:
    """Evaluate a candidate and add recovery-specific scores."""
    m = T.temporal_chain_measure(
        tc,
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        source_for_liveness=source_for_liveness,
        rng=rng,
    )
    min_residual = float(min(m["s1_residual_qcoords"], m["s2_residual_qcoords"]))
    res_norm = max(0.0, min(1.0, min_residual / max(float(tc.w), 1e-12)))
    transport = float(m["transport_mi_over_Hs2"])
    source = float(m["source_to_s2_mi_over_Hs2"])
    acc = float(m["s1_to_s2_best_accuracy"])
    # Accuracy is often high even for trivial labels, so it is only a small
    # contributor unless exact labels, transport, and liveness also exist.
    acc_norm = max(0.0, min(1.0, (acc - 1.0 / max(tc.q, 2)) / (1.0 - 1.0 / max(tc.q, 2))))
    exact_gate = bool(m["s1_residual_qcoords"] > 1e-9 and m["s2_residual_qcoords"] > 1e-9)
    live_transport = bool(exact_gate and transport >= transport_threshold and source >= source_threshold and acc >= accuracy_threshold)

    if fitness == "live_product":
        score = float(res_norm * max(transport, 0.0) * max(source, 0.0) * max(acc_norm, 1e-6))
    elif fitness == "transport_sum":
        # Mirrors the blindtemporalchain score; useful as a control.
        score = float(min_residual + 0.50 * transport + 0.50 * source + 1e-6 * acc)
    elif fitness == "live_sum":
        # Gate-biased but not fully discontinuous: exact labels matter, but
        # transport/source liveness can dominate among exact candidates.
        score = float(0.20 * res_norm + 0.35 * transport + 0.40 * source + 0.05 * acc_norm)
    else:
        raise ValueError("fitness must be live_sum, live_product, or transport_sum")

    out = dict(m)
    out.update(
        min_residual_qcoords=min_residual,
        residual_norm=res_norm,
        accuracy_norm=acc_norm,
        recovery_fitness=float(score),
        recovery_fitness_mode=str(fitness),
        exact_transport_gate=bool(exact_gate),
        selected_recovered_live_transport=bool(live_transport),
        transport_threshold=float(transport_threshold),
        source_threshold=float(source_threshold),
        accuracy_threshold=float(accuracy_threshold),
        seed_ensemble=str(tc.meta.get("seed_ensemble", tc.joint.meta.get("seed_ensemble", "unknown"))),
        seed_mutation_rate=float(tc.meta.get("seed_mutation_rate", tc.joint.meta.get("seed_mutation_rate", math.nan))),
    )
    return out


def _rank_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        float(row.get("recovery_fitness", 0.0)),
        float(row.get("source_to_s2_mi_over_Hs2", 0.0)),
        float(row.get("transport_mi_over_Hs2", 0.0)),
        float(row.get("min_residual_qcoords", 0.0)),
        float(row.get("s1_to_s2_best_accuracy", 0.0)),
    )


def run_seeded_recovery(
    kB: int,
    kM: int,
    kA: int,
    w: int,
    q: int,
    seed_ensemble: SeedEnsemble = "copy",
    seed_mutation_rate: float = 0.10,
    runs: int = 5,
    population: int = 20,
    generations: int = 60,
    fitness: FitnessMode = "live_sum",
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
        rng = np.random.default_rng(_stable_seed("seeded", base_seed, kB, kM, kA, w, q, seed_ensemble, seed_mutation_rate, run) % 2**32)
        pop = [
            make_seeded_candidate(kB, kM, kA, q, w, rng, seed_ensemble, seed_mutation_rate, seed_mutate_all_rules)
            for _ in range(int(population))
        ]
        for gen in range(int(generations) + 1):
            evals: list[tuple[float, dict, T.TemporalChain]] = []
            for ci, cand in enumerate(pop):
                erng = np.random.default_rng(_stable_seed("eval", base_seed, seed_ensemble, seed_mutation_rate, run, gen, ci) % 2**32)
                m = evaluate_recovery_candidate(
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
                m.update(
                    search_mode="seeded_recovery",
                    run=int(run),
                    generation=int(gen),
                    candidate=int(ci),
                    target_mode=str(target),
                    proposal_mode=str(proposal_mode),
                )
                rows.append(m)
                evals.append((float(m["recovery_fitness"]), m, cand))
            evals.sort(key=lambda t: _rank_key(t[1]), reverse=True)
            best = evals[0][1]
            saved.append((float(best["recovery_fitness"]), dict(best), B.clone_temporal_chain(evals[0][2])))
            if verbose:
                print(
                    "seeded-recovery "
                    f"ens={seed_ensemble} mu={seed_mutation_rate} run={run} gen={gen} "
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
                    new_pop.append(make_seeded_candidate(kB, kM, kA, q, w, rng, seed_ensemble, seed_mutation_rate, seed_mutate_all_rules))
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
                child.meta.update(seed_ensemble=str(seed_ensemble), seed_mutation_rate=float(seed_mutation_rate))
                child.joint.meta["seed_ensemble"] = str(seed_ensemble)
                child.joint.meta["seed_mutation_rate"] = float(seed_mutation_rate)
                new_pop.append(child)
            pop = new_pop
    return rows, saved


def run_seeded_temporal_recovery(
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2,),
    ws: Iterable[int] = (1,),
    q: int = 4,
    seed_ensembles: Iterable[SeedEnsemble] = ("copy", "permutive", "block", "canalizing", "random"),
    seed_mutation_rates: Iterable[float] = (0.05, 0.10),
    runs: int = 5,
    population: int = 20,
    generations: int = 60,
    fitness: FitnessMode = "live_sum",
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
                            rows, saved = run_seeded_recovery(
                                kB=int(kB), kM=int(kM), kA=int(kA), w=int(w), q=int(q),
                                seed_ensemble=ens,  # type: ignore[arg-type]
                                seed_mutation_rate=float(mu),
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
                                print(f"seeded-recovery complete ens={ens}, mu={mu}, kB={kB}, kM={kM}, kA={kA}, w={w}", flush=True)
    return pd.DataFrame(all_rows), all_saved


def analyze_seeded_recovery(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"verdict": "NO DATA"}
    rows = []
    for keys, g in df.groupby(["seed_ensemble", "seed_mutation_rate", "kB", "kM", "kA", "w"], dropna=False):
        ens, mu, kB, kM, kA, w = keys
        gen0 = g[g.generation == int(g.generation.min())]
        genf = g[g.generation == int(g.generation.max())]
        best0 = gen0.sort_values("recovery_fitness", ascending=False).head(1).iloc[0]
        bestf = genf.sort_values("recovery_fitness", ascending=False).head(1).iloc[0]
        rows.append({
            "seed_ensemble": str(ens),
            "seed_mutation_rate": float(mu),
            "kB": int(kB), "kM": int(kM), "kA": int(kA), "w": int(w),
            "initial_best_fitness": float(best0.recovery_fitness),
            "final_best_fitness": float(bestf.recovery_fitness),
            "fitness_improvement": float(bestf.recovery_fitness - best0.recovery_fitness),
            "initial_live_fraction": float(gen0.selected_recovered_live_transport.mean()),
            "final_live_fraction": float(genf.selected_recovered_live_transport.mean()),
            "initial_best_transport": float(best0.transport_mi_over_Hs2),
            "final_best_transport": float(bestf.transport_mi_over_Hs2),
            "initial_best_source_to_s2": float(best0.source_to_s2_mi_over_Hs2),
            "final_best_source_to_s2": float(bestf.source_to_s2_mi_over_Hs2),
            "initial_best_min_residual": float(best0.min_residual_qcoords),
            "final_best_min_residual": float(bestf.min_residual_qcoords),
            "final_best_live": bool(bestf.selected_recovered_live_transport),
        })
    summary_df = pd.DataFrame(rows)
    structured = summary_df[~summary_df.seed_ensemble.isin(["random", "constant"])]
    random = summary_df[summary_df.seed_ensemble == "random"]
    structured_recovered = float(structured.final_best_live.mean()) if len(structured) else 0.0
    random_recovered = float(random.final_best_live.mean()) if len(random) else 0.0
    structured_improvement = float(structured.fitness_improvement.mean()) if len(structured) else 0.0
    random_improvement = float(random.fitness_improvement.mean()) if len(random) else 0.0
    if structured_recovered > 0 and random_recovered == 0:
        verdict = "SEEDED TEMPORAL RECOVERY SIGNAL: selection repairs live transport inside structured basins but not random basins"
    elif structured_improvement > max(0.02, random_improvement + 0.02):
        verdict = "PARTIAL SEEDED RECOVERY SIGNAL: structured seeds improve more than random, but full live transport is rare"
    else:
        verdict = "NO SEEDED TEMPORAL RECOVERY SIGNAL in this regime"
    return {
        "verdict": verdict,
        "n_rows": int(len(df)),
        "structured_final_live_fraction": structured_recovered,
        "random_final_live_fraction": random_recovered,
        "mean_structured_fitness_improvement": structured_improvement,
        "mean_random_fitness_improvement": random_improvement,
        "by_seed": rows,
    }


def plot_seeded_recovery(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df is None or len(df) == 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # Best fitness over time by seed ensemble/mutation.
    for (ens, mu), g in df.groupby(["seed_ensemble", "seed_mutation_rate"], dropna=False):
        b = g.groupby("generation")["recovery_fitness"].max()
        ax[0].plot(b.index, b.values, marker="o", label=f"{ens} μ={mu:g}")
    ax[0].set_title("seeded recovery fitness")
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("best fitness")
    ax[0].legend(fontsize=7)
    # Final live fraction.
    last = df[df.generation == df.generation.max()]
    live = last.groupby(["seed_ensemble", "seed_mutation_rate"])["selected_recovered_live_transport"].mean()
    labels = [f"{a}\nμ={b:g}" for a, b in live.index]
    ax[1].bar(range(len(live)), live.values)
    ax[1].set_xticks(range(len(live)))
    ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("final live-transport fraction")
    ax[1].set_ylabel("fraction")
    # Source vs transport scatter.
    ax[2].scatter(df["source_to_s2_mi_over_Hs2"], df["transport_mi_over_Hs2"], s=10, alpha=0.5)
    ax[2].set_xlabel(r"$I(source;S_2)/H(S_2)$")
    ax[2].set_ylabel(r"$I(S_1;S_2)/H(S_2)$")
    ax[2].set_title("source-liveness vs transport")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _jsonify(x):
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if math.isnan(float(x)):
            return None
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x


def save_winners(path: str, saved: list[tuple[float, dict, T.TemporalChain]], top_n: int = 30, min_fitness: float = -math.inf) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    saved_sorted = sorted(saved, key=lambda t: _rank_key(t[1]), reverse=True)
    out = []
    for score, row, cand in saved_sorted:
        if float(score) < float(min_fitness):
            continue
        out.append({"score": float(score), "row": dict(row), "candidate": cand})
        if len(out) >= int(top_n):
            break
    with open(path, "wb") as f:
        pickle.dump({"kind": "seeded_temporal_recovery_winners", "winners": out}, f)


def _rank_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        float(row.get("selected_recovered_live_transport", False)),
        float(row.get("recovery_fitness", 0.0)),
        float(row.get("source_to_s2_mi_over_Hs2", 0.0)),
        float(row.get("transport_mi_over_Hs2", 0.0)),
        float(row.get("min_residual_qcoords", 0.0)),
    )


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Seeded temporal-chain recovery from noisy structured rules.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--ks-B", default="2")
    p.add_argument("--ks-M", default="2")
    p.add_argument("--ks-A", default="2")
    p.add_argument("--ws", default="1")
    p.add_argument("--seed-ensembles", default="copy,permutive,block,canalizing,random")
    p.add_argument("--seed-mutation-rates", default="0.05,0.1")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--fitness", choices=["live_sum", "live_product", "transport_sum"], default="live_sum")
    p.add_argument("--target", choices=["all", "interface", "source", "branch", "sink1", "sink2", "carriers"], default="interface")
    p.add_argument("--proposal-mode", choices=["entry", "table", "mixed"], default="mixed")
    p.add_argument("--entry-rate", type=float, default=0.04)
    p.add_argument("--table-rate", type=float, default=0.03)
    p.add_argument("--n-table-mutations", type=int, default=1)
    p.add_argument("--elite-fraction", type=float, default=0.25)
    p.add_argument("--reseed-fraction", type=float, default=0.0)
    p.add_argument("--seed-mutate-interface-only", action="store_true")
    p.add_argument("--initial-mode", choices=["source_all", "source_random", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--source-for-liveness", choices=["initial", "postB"], default="postB")
    p.add_argument("--transport-threshold", type=float, default=0.80)
    p.add_argument("--source-threshold", type=float, default=0.80)
    p.add_argument("--accuracy-threshold", type=float, default=0.95)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out", default="example_results/seeded_temporal_recovery.csv")
    p.add_argument("--plot", default="")
    p.add_argument("--save-winners", default="")
    p.add_argument("--save-winner-top-n", type=int, default=30)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    df, saved = run_seeded_temporal_recovery(
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
    summary = analyze_seeded_recovery(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)
    if args.plot:
        plot_seeded_recovery(df, args.plot)
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
