"""
closedseededrecovery.py -- seeded recovery for closed SCC C2 media.

The blind closed-SCC search is deliberately hard: random closed recurrent
systems have active flux but almost no schedule-consistent binary quotient.  The
control sweep shows that closed C2 quotients and recurrent C2 media exist in
structured ensembles.  This module asks the intermediate question:

    once the search is seeded near a closed C2 quotient/medium, can local
    consistency selection retain or repair it?

The default fitness rewards the same quotient-level properties used by
closedplaquettelattice.py: binary link quotient determinism, schedule
consistency, and nontrivial quotient entropy.  Recurrent flux entropy remains a
post-hoc diagnostic by default.  An optional ``--fitness medium`` mode adds a
small recurrent-flux term for stress testing, but the publication-clean mode is
``--fitness quotient``.

Recommended first run
---------------------
python -m relgauge.closedseededrecovery 4 ^
  --sizes 2x2 ^
  --seed-ensembles copy,c2xor,random ^
  --seed-mutation-rates 0,0.02,0.05 ^
  --runs 5 ^
  --population 24 ^
  --generations 100 ^
  --fitness quotient ^
  --n-initial 4096 ^
  --horizon 24 ^
  --schedule-samples 8 ^
  --out example_results/closed_seeded_recovery_q4_2x2.csv ^
  --plot example_results/fig_closed_seeded_recovery_q4_2x2.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import closedplaquettelattice as CL
from . import core as C


def _parse_size(s: str) -> tuple[int, int]:
    if "x" in str(s).lower():
        a, b = str(s).lower().split("x", 1)
        return int(a), int(b)
    n = int(s)
    return n, n


def _parse_list(s: str, typ=str):
    if not s:
        return []
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def _rule_signature(lat: CL.ClosedPlaquetteLattice) -> tuple:
    return tuple(tuple(int(x) for x in r.tolist()) for r in lat.sys.rules)


def _copy_lattice(lat: CL.ClosedPlaquetteLattice) -> CL.ClosedPlaquetteLattice:
    rules = [np.asarray(r, dtype=np.int64).copy() for r in lat.sys.rules]
    sys = C.RelationalSystem(lat.sys.k, lat.sys.q, list(lat.sys.preds), rules, meta=dict(lat.sys.meta))
    return CL.ClosedPlaquetteLattice(
        sys=sys,
        q=int(lat.q),
        rows=int(lat.rows),
        cols=int(lat.cols),
        H=lat.H.copy(),
        V=lat.V.copy(),
        canonical_schedule=tuple(lat.canonical_schedule),
        meta=dict(lat.meta),
    )


def _mutate_lattice(lat: CL.ClosedPlaquetteLattice, rng: np.random.Generator, entry_rate: float, table_rate: float) -> CL.ClosedPlaquetteLattice:
    # Use the implementation shipped with closedplaquettelattice when present;
    # keep a fallback so this module remains robust if the private helper is
    # renamed later.
    if hasattr(CL, "_mutate_lattice"):
        return CL._mutate_lattice(lat, rng, entry_rate=entry_rate, table_rate=table_rate)  # type: ignore[attr-defined]
    child = _copy_lattice(lat)
    q = int(child.q)
    for v, rule in enumerate(child.sys.rules):
        rule = np.asarray(rule, dtype=np.int64).copy()
        if rng.random() < float(table_rate):
            rule = rng.integers(0, q, size=rule.size, dtype=np.int64)
        else:
            mask = rng.random(rule.size) < float(entry_rate)
            if np.any(mask):
                jumps = rng.integers(1, q, size=int(mask.sum()), dtype=np.int64)
                rule[mask] = (rule[mask] + jumps) % q
        child.sys.rules[v] = rule
    return child


def _load_seed_lattices(path: str, top_n: int | None = None) -> list[CL.ClosedPlaquetteLattice]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        obj = [obj]
    out: list[CL.ClosedPlaquetteLattice] = []
    for item in obj:
        lat = None
        if isinstance(item, CL.ClosedPlaquetteLattice):
            lat = item
        elif isinstance(item, dict):
            if isinstance(item.get("lattice"), CL.ClosedPlaquetteLattice):
                lat = item["lattice"]
            elif isinstance(item.get("lat"), CL.ClosedPlaquetteLattice):
                lat = item["lat"]
        if lat is not None:
            out.append(_copy_lattice(lat))
        if top_n is not None and len(out) >= int(top_n):
            break
    if not out:
        raise ValueError(f"no ClosedPlaquetteLattice objects found in {path!r}")
    return out


def _fitness(row: dict, mode: str = "quotient") -> float:
    """Seeded recovery score.

    ``quotient`` is the clean mode: it rewards closed quotient consistency only.
    ``medium`` adds a bounded recurrent-flux entropy term and is meant as a
    stress test, not as the purity-preserving headline mode.
    """
    base = float(row.get("quotient_score", 0.0))
    if bool(row.get("closed_c2_quotient", False)):
        base += 0.25
    if mode == "medium":
        # Maximum flux entropy is the number of plaquettes.  Keep the term small
        # and gated by closed C2 validity so entropy without consistency is not
        # rewarded.
        maxH = max(1.0, float(row.get("n_plaquettes", 1)))
        if bool(row.get("closed_c2_quotient", False)):
            base += 0.20 * min(1.0, float(row.get("recurrent_flux_entropy_bits", 0.0)) / maxH)
    elif mode != "quotient":
        raise ValueError("fitness must be 'quotient' or 'medium'")
    return float(base)


def _make_seed(
    q: int,
    size: tuple[int, int],
    ensemble: str,
    mutation_rate: float,
    rng: np.random.Generator,
) -> CL.ClosedPlaquetteLattice:
    rows, cols = size
    return CL.make_closed_lattice(int(q), int(rows), int(cols), rng, ensemble=ensemble, mutation_rate=float(mutation_rate))


def run_seeded_closed_recovery(
    q: int = 4,
    size: tuple[int, int] = (2, 2),
    seed_ensembles: Iterable[str] = ("copy", "c2xor", "random"),
    seed_mutation_rates: Iterable[float] = (0.0, 0.02, 0.05),
    runs: int = 5,
    population: int = 24,
    generations: int = 100,
    n_initial: int = 4096,
    horizon: int = 24,
    schedule_samples: int = 8,
    entry_rate: float = 0.06,
    table_rate: float = 0.08,
    fitness_mode: str = "quotient",
    seed_winners: str | None = None,
    top_n: int = 20,
    base_seed: int = 0,
    save_winners: str | None = None,
    save_winner_top_n: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    rows_out: list[dict] = []
    winner_pool: list[tuple[float, CL.ClosedPlaquetteLattice, dict]] = []
    seed_pool: list[tuple[str, float, int, CL.ClosedPlaquetteLattice]] = []
    R, Cc = size

    if seed_winners:
        loaded = _load_seed_lattices(seed_winners, top_n=top_n)
        for i, lat in enumerate(loaded):
            seed_pool.append(("winner", 0.0, i, lat))
    else:
        for ens in seed_ensembles:
            for mu in seed_mutation_rates:
                for i in range(int(runs)):
                    seed = (int(base_seed) * 1000003 + int(q) * 9176 + int(R) * 313 + int(Cc) * 571 + hash(str(ens)) % 1000 * 37 + int(round(float(mu) * 10000)) * 11 + i) % 2**32
                    rng = np.random.default_rng(seed)
                    lat = _make_seed(q, size, str(ens), float(mu), rng)
                    seed_pool.append((str(ens), float(mu), i, lat))

    for seed_ensemble, seed_mu, run_id, seed_lat in seed_pool:
        rng_seed = (int(base_seed) * 99991 + int(q) * 1237 + int(run_id) * 104729 + hash((seed_ensemble, round(seed_mu, 6))) % 100000) % 2**32
        rng = np.random.default_rng(rng_seed)
        pop: list[CL.ClosedPlaquetteLattice] = [_copy_lattice(seed_lat)]
        # Make the first generation a local cloud around the seed.  The seed
        # itself is kept as an elite so retention can be measured directly.
        while len(pop) < int(population):
            pop.append(_mutate_lattice(seed_lat, rng, entry_rate=entry_rate, table_rate=table_rate))
        for gen in range(int(generations) + 1):
            scored: list[tuple[float, CL.ClosedPlaquetteLattice, dict]] = []
            for ci, lat in enumerate(pop):
                row = CL.measure_closed_lattice(
                    lat,
                    n_initial=int(n_initial),
                    horizon=int(horizon),
                    schedule_samples=int(schedule_samples),
                    rng=rng,
                )
                fit = _fitness(row, fitness_mode)
                row.update(
                    search_mode="seeded",
                    seed_ensemble=str(seed_ensemble),
                    seed_mutation_rate=float(seed_mu),
                    seed_index=int(run_id),
                    run=int(run_id),
                    generation=int(gen),
                    candidate=int(ci),
                    recovery_fitness=float(fit),
                    fitness_mode=str(fitness_mode),
                )
                rows_out.append(row)
                scored.append((float(fit), lat, row))
                winner_pool.append((float(fit), lat, row))
            scored.sort(key=lambda z: z[0], reverse=True)
            best_row = scored[0][2]
            if verbose:
                print(
                    f"closed-seeded seed={seed_ensemble} mu={seed_mu} run={run_id} gen={gen} "
                    f"best={scored[0][0]:.4f} c2={best_row['closed_c2_quotient']} "
                    f"medium={best_row['closed_medium_candidate']} recH={best_row['recurrent_flux_entropy_bits']:.3f} "
                    f"det={best_row['quotient_determinism_accuracy']:.3f}",
                    flush=True,
                )
            if gen == int(generations):
                break
            elites = [x[1] for x in scored[: max(1, int(population) // 4)]]
            new_pop: list[CL.ClosedPlaquetteLattice] = [_copy_lattice(e) for e in elites]
            while len(new_pop) < int(population):
                parent = elites[int(rng.integers(0, len(elites)))]
                child = _mutate_lattice(parent, rng, entry_rate=entry_rate, table_rate=table_rate)
                new_pop.append(child)
            pop = new_pop

    if save_winners:
        os.makedirs(os.path.dirname(save_winners) or ".", exist_ok=True)
        winner_pool.sort(key=lambda z: z[0], reverse=True)
        packed = []
        seen = set()
        for fit, lat, row in winner_pool:
            key = _rule_signature(lat)
            if key in seen:
                continue
            seen.add(key)
            packed.append(dict(fitness=float(fit), row=dict(row), lattice=_copy_lattice(lat)))
            if len(packed) >= int(save_winner_top_n):
                break
        with open(save_winners, "wb") as f:
            pickle.dump(packed, f)

    return pd.DataFrame(rows_out)


def analyze_seeded_recovery(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    maxg = int(df["generation"].max()) if "generation" in df.columns else 0
    initial = df[df["generation"] == 0]
    final = df[df["generation"] == maxg]
    # Best per seed/run at initial/final generation.
    def best_per_run(g: pd.DataFrame) -> pd.DataFrame:
        if g.empty:
            return g
        idx = g.groupby(["seed_ensemble", "seed_mutation_rate", "run"])["recovery_fitness"].idxmax()
        return g.loc[idx]

    bi = best_per_run(initial)
    bf = best_per_run(final)
    all_best_idx = df["recovery_fitness"].idxmax()
    all_best = df.loc[all_best_idx]

    nonrandom_final = bf[bf.seed_ensemble != "random"] if "seed_ensemble" in bf else bf
    random_final = bf[bf.seed_ensemble == "random"] if "seed_ensemble" in bf else bf.iloc[0:0]
    final_closed = float(bf.closed_c2_quotient.mean()) if len(bf) else 0.0
    final_medium = float(bf.closed_medium_candidate.mean()) if len(bf) else 0.0
    structured_final_closed = float(nonrandom_final.closed_c2_quotient.mean()) if len(nonrandom_final) else math.nan
    structured_final_medium = float(nonrandom_final.closed_medium_candidate.mean()) if len(nonrandom_final) else math.nan
    random_final_closed = float(random_final.closed_c2_quotient.mean()) if len(random_final) else math.nan
    random_final_medium = float(random_final.closed_medium_candidate.mean()) if len(random_final) else math.nan
    initial_closed = float(bi.closed_c2_quotient.mean()) if len(bi) else 0.0
    initial_medium = float(bi.closed_medium_candidate.mean()) if len(bi) else 0.0

    if (not math.isnan(structured_final_medium)) and structured_final_medium > max(0.0, random_final_medium if not math.isnan(random_final_medium) else 0.0) and structured_final_medium > 0:
        verdict = "SEEDED CLOSED-SCC MEDIUM RECOVERY: closed recurrent C2 media are retained/repaired from structured seeds"
    elif (not math.isnan(structured_final_closed)) and structured_final_closed > 0:
        verdict = "SEEDED CLOSED-SCC C2 QUOTIENT RECOVERY: binary quotient survives seeded closure but recurrent medium is not robust"
    else:
        verdict = "NO SEEDED CLOSED-SCC RECOVERY SIGNAL in this regime"

    by_seed = []
    if "seed_ensemble" in df.columns:
        for (ens, mu), g in df.groupby(["seed_ensemble", "seed_mutation_rate"]):
            gi = best_per_run(g[g.generation == g.generation.min()])
            gf = best_per_run(g[g.generation == g.generation.max()])
            by_seed.append(dict(
                seed_ensemble=str(ens),
                seed_mutation_rate=float(mu),
                n_runs=int(g["run"].nunique()),
                initial_closed_c2_fraction=float(gi.closed_c2_quotient.mean()) if len(gi) else math.nan,
                final_closed_c2_fraction=float(gf.closed_c2_quotient.mean()) if len(gf) else math.nan,
                initial_closed_medium_fraction=float(gi.closed_medium_candidate.mean()) if len(gi) else math.nan,
                final_closed_medium_fraction=float(gf.closed_medium_candidate.mean()) if len(gf) else math.nan,
                final_mean_recurrent_flux_entropy=float(gf.recurrent_flux_entropy_bits.mean()) if len(gf) else math.nan,
                final_mean_link_schedule=float(gf.link_schedule_consistency.mean()) if len(gf) else math.nan,
                final_mean_flux_schedule=float(gf.flux_schedule_consistency.mean()) if len(gf) else math.nan,
                final_mean_det_accuracy=float(gf.quotient_determinism_accuracy.mean()) if len(gf) else math.nan,
                final_mean_fitness=float(gf.recovery_fitness.mean()) if len(gf) else math.nan,
            ))

    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        n_seed_runs=int(bf.shape[0]),
        initial_closed_c2_fraction=initial_closed,
        initial_closed_medium_fraction=initial_medium,
        final_closed_c2_fraction=final_closed,
        final_closed_medium_fraction=final_medium,
        structured_final_closed_c2_fraction=structured_final_closed,
        structured_final_closed_medium_fraction=structured_final_medium,
        random_final_closed_c2_fraction=random_final_closed,
        random_final_closed_medium_fraction=random_final_medium,
        all_time_best_fitness=float(all_best["recovery_fitness"]),
        all_time_best_seed=str(all_best.get("seed_ensemble", "unknown")),
        all_time_best_closed_c2=bool(all_best["closed_c2_quotient"]),
        all_time_best_closed_medium=bool(all_best["closed_medium_candidate"]),
        all_time_best_recurrent_flux_entropy=float(all_best["recurrent_flux_entropy_bits"]),
        all_time_best_link_schedule=float(all_best["link_schedule_consistency"]),
        all_time_best_flux_schedule=float(all_best["flux_schedule_consistency"]),
        by_seed=by_seed,
    )


def plot_seeded_recovery(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    g = df.groupby("generation")
    ax[0].plot(g["recovery_fitness"].max(), marker="o", label="global best")
    ax[0].plot(g["recovery_fitness"].mean(), marker="o", label="mean")
    ax[0].set_title("seeded closed-SCC fitness")
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("fitness")
    ax[0].legend()

    ax[1].plot(g["closed_c2_quotient"].mean(), marker="o", label="C2 quotient")
    ax[1].plot(g["closed_medium_candidate"].mean(), marker="o", label="recurrent medium")
    ax[1].set_title("closed recovery over time")
    ax[1].set_xlabel("generation")
    ax[1].set_ylabel("fraction")
    ax[1].legend()

    # Final best per run, colored by seed ensemble when possible.
    maxg = int(df["generation"].max())
    final = df[df.generation == maxg]
    idx = final.groupby(["seed_ensemble", "seed_mutation_rate", "run"])["recovery_fitness"].idxmax()
    bf = final.loc[idx]
    for ens, gg in bf.groupby("seed_ensemble"):
        ax[2].scatter(gg["link_schedule_consistency"], gg["recurrent_flux_entropy_bits"], label=str(ens), alpha=0.8)
    ax[2].set_title("final schedule consistency vs flux entropy")
    ax[2].set_xlabel("link schedule consistency")
    ax[2].set_ylabel("recurrent flux entropy")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Seeded recovery for closed SCC C2 quotient/media.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--sizes", default="2x2")
    p.add_argument("--seed-ensembles", default="copy,c2xor,random")
    p.add_argument("--seed-mutation-rates", default="0,0.02,0.05")
    p.add_argument("--seed-winners", default=None)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--fitness", choices=["quotient", "medium"], default="quotient")
    p.add_argument("--n-initial", type=int, default=4096)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--schedule-samples", type=int, default=8)
    p.add_argument("--entry-rate", type=float, default=0.06)
    p.add_argument("--table-rate", type=float, default=0.08)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--save-winners", default=None)
    p.add_argument("--save-winner-top-n", type=int, default=50)
    p.add_argument("--out", default="example_results/closed_seeded_recovery.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    sizes = [_parse_size(x) for x in _parse_list(args.sizes, str)]
    if len(sizes) != 1:
        raise SystemExit("closedseededrecovery currently accepts exactly one --sizes entry")
    df = run_seeded_closed_recovery(
        q=int(args.q),
        size=sizes[0],
        seed_ensembles=_parse_list(args.seed_ensembles, str),
        seed_mutation_rates=_parse_list(args.seed_mutation_rates, float),
        runs=int(args.runs),
        population=int(args.population),
        generations=int(args.generations),
        n_initial=int(args.n_initial),
        horizon=int(args.horizon),
        schedule_samples=int(args.schedule_samples),
        entry_rate=float(args.entry_rate),
        table_rate=float(args.table_rate),
        fitness_mode=str(args.fitness),
        seed_winners=args.seed_winners,
        top_n=int(args.top_n),
        base_seed=int(args.base_seed),
        save_winners=args.save_winners,
        save_winner_top_n=int(args.save_winner_top_n),
        verbose=not args.quiet,
    )
    summary = analyze_seeded_recovery(df)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_seeded_recovery(df, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
