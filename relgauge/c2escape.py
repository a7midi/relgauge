"""
c2escape.py -- can a blind-selected C2 shared-square gauge sector escape to larger S?

The blind shared-corner square search repeatedly finds a binary transported
holonomy sector (|S|=2, generated group C2) from random starts.  This module
loads those saved C2 winners and continues selection from inside that basin.

Question
--------
Is C2 merely the first foothold in rule space, or is it a stable local attractor
of the consistency-selection landscape?

For q=4,w=1 the relevant residuals are:
    |S|=2  -> log_4 2 = 0.5      (C2/binary sector)
    |S|=3  -> log_4 3 ~= 0.7925  (escape to ternary common factor)
    |S|=4  -> log_4 4 = 1.0      (full alphabet transport)

The default fitness is the same transport-only objective used by
blindsharedsquare after pretraining: it rewards live/bijective transport and
larger min-residual, but does not reward any named group or holonomy type.
Holonomy/group structure are reported post hoc.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from typing import Iterable

import numpy as np
import pandas as pd

from . import blindsharedsquare as BS
from . import sharedsquareholonomy as SH


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


def _load_seed_winners(path: str, top_n: int | None = None, require_valid: bool = True) -> list[tuple[float, SH.SharedSquare, dict]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    winners = data.get("winners", []) if isinstance(data, dict) else []
    out: list[tuple[float, SH.SharedSquare, dict]] = []
    for item in winners:
        if not isinstance(item, dict) or "square" not in item:
            continue
        metrics = dict(item.get("metrics", {}))
        if require_valid and not bool(metrics.get("valid_square", False)):
            continue
        fit = _safe_float(item.get("fitness", metrics.get("blind_fitness", 0.0)), 0.0)
        out.append((fit, item["square"], metrics))
    out.sort(key=lambda t: (
        bool(t[2].get("valid_square", False)),
        float(t[2].get("min_residual_qcoords", 0.0) or 0.0),
        bool(t[2].get("nontrivial_path_holonomy", False)),
        float(t[2].get("blind_fitness", t[0]) or 0.0),
    ), reverse=True)
    if top_n is not None and int(top_n) > 0:
        out = out[:int(top_n)]
    if not out:
        raise ValueError(f"no usable seed winners loaded from {path!r}")
    return out


def _make_initial_population(
    seed_pool: list[tuple[float, SH.SharedSquare, dict]],
    population: int,
    rng: np.random.Generator,
    target: str,
    seed_jitter_entry_rate: float = 0.01,
    seed_jitter_table_rate: float = 0.01,
) -> list[SH.SharedSquare]:
    pop: list[SH.SharedSquare] = []
    # Keep several exact seed elites unchanged.
    n_elite = min(len(seed_pool), max(1, int(population) // 4))
    for _, sq, _ in seed_pool[:n_elite]:
        pop.append(BS.clone_square(sq))
    # Fill with lightly jittered descendants of saved winners.
    while len(pop) < int(population):
        _, parent, _ = seed_pool[int(rng.integers(0, len(seed_pool)))]
        child = BS.clone_square(parent)
        if float(seed_jitter_entry_rate) > 0 and float(seed_jitter_table_rate) > 0:
            child = BS.mutate_square(
                child,
                rng,
                entry_rate=float(seed_jitter_entry_rate),
                table_rate=float(seed_jitter_table_rate),
                target=target,
                force_one=True,
            )
        pop.append(child)
    return pop


def evaluate_escape_candidate(
    sq: SH.SharedSquare,
    rng: np.random.Generator,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    fitness: str = "transport",
) -> dict:
    # Use blindsharedsquare's evaluator to keep semantics identical.
    if fitness not in {"transport", "holonomy", "pretrain"}:
        fitness = "transport"
    row = BS.evaluate_candidate(
        sq,
        rng,
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        fitness=fitness,  # no two-stage here: this is the second stage
        generation=0,
        pretrain_generations=0,
    )
    q = int(row.get("q", sq.q))
    mclasses = _min_classes(row)
    row["min_shared_classes"] = int(mclasses)
    row["escape_to_3_classes"] = bool(mclasses >= 3)
    row["escape_to_full_classes"] = bool(mclasses >= q)
    row["c2_retained"] = bool(mclasses == 2 and str(row.get("generated_group_name", "")) == "C2")
    row["higher_than_c2"] = bool(mclasses > 2)
    return row


def _select_parent(scored_pop: list[tuple[float, SH.SharedSquare, dict]], rng: np.random.Generator) -> SH.SharedSquare:
    # Tournament among top half.
    top = scored_pop[:max(2, len(scored_pop)//2)]
    idx = rng.choice(len(top), size=min(3, len(top)), replace=False)
    return BS.clone_square(max((top[int(i)] for i in idx), key=lambda t: t[0])[1])


def run_escape_evolution(
    seed_pool: list[tuple[float, SH.SharedSquare, dict]],
    runs: int = 10,
    population: int = 40,
    generations: int = 150,
    fitness: str = "transport",
    target: str = "interface",
    proposal_mode: str = "mutate",
    entry_rate: float = 0.04,
    table_rate: float = 0.05,
    seed_jitter_entry_rate: float = 0.01,
    seed_jitter_table_rate: float = 0.01,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[float, SH.SharedSquare, dict]]]:
    rows: list[dict] = []
    winner_pool: list[tuple[float, SH.SharedSquare, dict]] = []
    if not seed_pool:
        raise ValueError("seed_pool is empty")
    q = int(seed_pool[0][1].q); w = int(seed_pool[0][1].w)

    # Evaluate raw seeds once for baseline.
    for si, (_, sq, sm) in enumerate(seed_pool):
        rng = np.random.default_rng((int(base_seed) + 13 * si + 7777) % 2**32)
        row = evaluate_escape_candidate(sq, rng, initial_mode, n_random_initial, fitness=fitness)
        row.update(search_mode="seed", run=-1, generation=0, candidate=int(si), seed_index=int(si), seed_metric_min_residual=float(sm.get("min_residual_qcoords", math.nan)))
        rows.append(row)
        winner_pool.append((float(row["blind_fitness"]), BS.clone_square(sq), row))

    for run in range(int(runs)):
        rng = np.random.default_rng((int(base_seed) + 1000003 + 7919 * run + 17 * q + w) % 2**32)
        pop = _make_initial_population(seed_pool, population, rng, target, seed_jitter_entry_rate, seed_jitter_table_rate)
        scored: list[tuple[float, SH.SharedSquare, dict]] = []
        for gen in range(int(generations) + 1):
            scored = []
            for ci, sq in enumerate(pop):
                erng = np.random.default_rng(rng.integers(0, 2**32 - 1, dtype=np.uint32).item())
                row = evaluate_escape_candidate(sq, erng, initial_mode, n_random_initial, fitness=fitness)
                row.update(
                    search_mode="escape",
                    run=int(run),
                    generation=int(gen),
                    candidate=int(ci),
                    target_mode=target,
                    proposal_mode=proposal_mode,
                    entry_rate=float(entry_rate),
                    table_rate=float(table_rate),
                    fitness_mode=fitness,
                )
                rows.append(row); scored.append((float(row["blind_fitness"]), sq, row))
            scored.sort(key=lambda t: t[0], reverse=True)
            winner_pool.extend(scored[:min(5, len(scored))])
            best = scored[0][2]
            if verbose:
                print(
                    "c2-escape "
                    f"run={run} gen={gen} best={best.get('blind_fitness'):.4f} "
                    f"classes={best.get('min_shared_classes')} minres={best.get('min_residual_qcoords'):.4f} "
                    f"src={best.get('source_liveness'):.4f} trans={best.get('min_edge_transport'):.4f} "
                    f"group={best.get('generated_group_name')} escape3={bool(best.get('escape_to_3_classes'))}",
                    flush=True,
                )
            if gen == int(generations):
                break
            n_elite = max(2, int(population)//6)
            next_pop = [BS.clone_square(t[1]) for t in scored[:n_elite]]
            while len(next_pop) < int(population):
                if proposal_mode == "random":
                    child = BS.random_square(q, w, rng)
                elif proposal_mode == "mixed" and rng.random() < 0.05:
                    child = BS.random_square(q, w, rng)
                else:
                    parent = _select_parent(scored, rng)
                    child = BS.mutate_square(parent, rng, entry_rate=float(entry_rate), table_rate=float(table_rate), target=target, force_one=True)
                next_pop.append(child)
            pop = next_pop
    return pd.DataFrame(rows), winner_pool


def run_c2_escape(
    seed_winners: str,
    top_n: int = 30,
    runs: int = 10,
    population: int = 40,
    generations: int = 150,
    fitness: str = "transport",
    target: str = "interface",
    proposal_mode: str = "mutate",
    entry_rate: float = 0.04,
    table_rate: float = 0.05,
    seed_jitter_entry_rate: float = 0.01,
    seed_jitter_table_rate: float = 0.01,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[float, SH.SharedSquare, dict]]]:
    seed_pool = _load_seed_winners(seed_winners, top_n=top_n, require_valid=True)
    return run_escape_evolution(
        seed_pool,
        runs=runs,
        population=population,
        generations=generations,
        fitness=fitness,
        target=target,
        proposal_mode=proposal_mode,
        entry_rate=entry_rate,
        table_rate=table_rate,
        seed_jitter_entry_rate=seed_jitter_entry_rate,
        seed_jitter_table_rate=seed_jitter_table_rate,
        initial_mode=initial_mode,
        n_random_initial=n_random_initial,
        base_seed=base_seed,
        verbose=verbose,
    )


def _best_by_run_generation(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df.search_mode == "escape"].copy()
    if ev.empty:
        return ev
    idx = ev.groupby(["run", "generation"])["blind_fitness"].idxmax()
    return ev.loc[idx].copy()


def analyze_c2_escape(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    seeds = df[df.search_mode == "seed"].copy()
    best = _best_by_run_generation(df)
    final = best.loc[best.groupby("run")["generation"].idxmax()] if len(best) else best
    valid = df[df.valid_square.astype(bool)] if "valid_square" in df else df.iloc[0:0]
    all_time = df.loc[df["blind_fitness"].idxmax()].to_dict() if len(df) else {}
    q = int(df.q.dropna().iloc[0]) if "q" in df and len(df.q.dropna()) else 4
    log2 = float(math.log(2, q)) if q > 1 else math.nan
    log3 = float(math.log(3, q)) if q > 2 else math.inf

    seed_max_classes = int(seeds.min_shared_classes.max()) if len(seeds) else 0
    final_max_classes = int(final.min_shared_classes.max()) if len(final) else 0
    all_max_classes = int(df.min_shared_classes.max()) if "min_shared_classes" in df else 0
    final_escape3 = float(final.escape_to_3_classes.mean()) if len(final) else math.nan
    final_full = float(final.escape_to_full_classes.mean()) if len(final) else math.nan
    final_c2 = float(final.c2_retained.mean()) if len(final) else math.nan
    any_escape3 = bool(df.escape_to_3_classes.any()) if "escape_to_3_classes" in df else False
    any_full = bool(df.escape_to_full_classes.any()) if "escape_to_full_classes" in df else False

    if final_escape3 > 0.25 or any_full:
        verdict = "C2 ESCAPE SIGNAL: continued selection reaches larger transported alphabets"
    elif any_escape3:
        verdict = "RARE C2 ESCAPE: isolated candidates reach |S|>=3 but it is not stable across runs"
    else:
        verdict = "C2 STABLE BASIN: no escape to larger |S| found under this local search"

    by_gen = []
    if len(best):
        for gen, g in best.groupby("generation"):
            by_gen.append(dict(
                generation=int(gen),
                best_fitness=float(g.blind_fitness.max()),
                mean_best_fitness=float(g.blind_fitness.mean()),
                max_classes=int(g.min_shared_classes.max()),
                mean_classes=float(g.min_shared_classes.mean()),
                escape3_run_fraction=float(g.escape_to_3_classes.mean()),
                full_run_fraction=float(g.escape_to_full_classes.mean()),
                c2_run_fraction=float(g.c2_retained.mean()),
                valid_run_fraction=float(g.valid_square.mean()),
                holonomy_run_fraction=float(g[g.valid_square.astype(bool)].nontrivial_path_holonomy.mean()) if len(g[g.valid_square.astype(bool)]) else 0.0,
            ))

    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        q=q,
        c2_residual_qcoords=log2,
        ternary_residual_qcoords=log3 if math.isfinite(log3) else None,
        seed_count=int(len(seeds)),
        seed_max_classes=seed_max_classes,
        seed_escape3_fraction=float(seeds.escape_to_3_classes.mean()) if len(seeds) else math.nan,
        seed_c2_fraction=float(seeds.c2_retained.mean()) if len(seeds) else math.nan,
        final_run_count=int(len(final)),
        final_max_classes=final_max_classes,
        all_time_max_classes=all_max_classes,
        final_escape3_fraction=final_escape3,
        final_full_fraction=final_full,
        final_c2_fraction=final_c2,
        any_escape_to_3=any_escape3,
        any_escape_to_full=any_full,
        all_time_best_fitness=float(all_time.get("blind_fitness", math.nan)),
        all_time_best_classes=int(all_time.get("min_shared_classes", 0) or 0),
        all_time_best_min_residual=float(all_time.get("min_residual_qcoords", math.nan)),
        all_time_best_group=str(all_time.get("generated_group_name", "")),
        all_time_best_holonomy=bool(all_time.get("nontrivial_path_holonomy", False)),
        generated_group_counts={str(k): int(v) for k, v in (valid.generated_group_name.value_counts().items() if len(valid) else [])},
        delta_type_counts={str(k): int(v) for k, v in (valid.delta_type.value_counts().items() if len(valid) else [])},
        best_by_generation=by_gen,
    )


def save_winners(path: str, winners: list[tuple[float, SH.SharedSquare, dict]], top_n: int = 50) -> None:
    ranked = sorted(winners, key=lambda t: (
        int(t[2].get("min_shared_classes", 0) or 0),
        bool(t[2].get("escape_to_full_classes", False)),
        bool(t[2].get("escape_to_3_classes", False)),
        bool(t[2].get("valid_square", False)),
        float(t[2].get("blind_fitness", t[0]) or 0.0),
    ), reverse=True)
    out = []
    for fit, sq, row in ranked[:int(top_n)]:
        out.append(dict(fitness=float(fit), metrics=dict(row), square=sq))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(kind="c2_escape_winners", winners=out), f)


def plot_c2_escape(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    best = _best_by_run_generation(df)
    if len(best):
        g = best.groupby("generation")
        gens = list(g.groups)
        ax[0].plot(gens, [g.get_group(k).blind_fitness.max() for k in gens], "o-", label="global best")
        ax[0].plot(gens, [g.get_group(k).blind_fitness.mean() for k in gens], "o-", label="mean run-best")
        ax[0].set_title("C2 escape fitness")
        ax[0].set_xlabel("generation"); ax[0].set_ylabel("fitness")
        ax[0].legend(fontsize=8)
        ax[1].plot(gens, [g.get_group(k).min_shared_classes.max() for k in gens], "o-", label="max |S|")
        ax[1].plot(gens, [g.get_group(k).min_shared_classes.mean() for k in gens], "o-", label="mean run-best |S|")
        ax[1].axhline(2, linestyle="--", linewidth=1, label="C2")
        ax[1].axhline(3, linestyle=":", linewidth=1, label="escape |S|=3")
        ax[1].set_title("escape from binary sector")
        ax[1].set_xlabel("generation"); ax[1].set_ylabel("min shared classes")
        ax[1].legend(fontsize=8)
        ax[2].plot(gens, [g.get_group(k).escape_to_3_classes.mean() for k in gens], "o-", label="|S|>=3")
        ax[2].plot(gens, [g.get_group(k).escape_to_full_classes.mean() for k in gens], "o-", label="|S|=q")
        ax[2].plot(gens, [g.get_group(k).c2_retained.mean() for k in gens], "o-", label="C2 retained")
        ax[2].set_title("run-best fractions")
        ax[2].set_xlabel("generation"); ax[2].set_ylabel("fraction")
        ax[2].legend(fontsize=8)
    fig.tight_layout(); os.makedirs(os.path.dirname(path) or ".", exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Test whether blind-selected C2 shared-square winners escape to larger transported alphabets.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--seed-winners", required=True, help="Pickle from blindsharedsquare --save-winners")
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--fitness", choices=["transport", "holonomy", "pretrain"], default="transport")
    p.add_argument("--target", choices=["interface", "all"], default="interface")
    p.add_argument("--proposal-mode", choices=["mutate", "mixed", "random"], default="mutate")
    p.add_argument("--entry-rate", type=float, default=0.04)
    p.add_argument("--table-rate", type=float, default=0.05)
    p.add_argument("--seed-jitter-entry-rate", type=float, default=0.01)
    p.add_argument("--seed-jitter-table-rate", type=float, default=0.01)
    p.add_argument("--initial-mode", choices=["source_all", "source_random", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--save-winners", default=None)
    p.add_argument("--save-winner-top-n", type=int, default=50)
    p.add_argument("--out", default="example_results/c2_escape.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    # q is not used to construct; the winners determine q.  Keep it as a CLI
    # positional for consistency and to make command logs self-documenting.
    df, winners = run_c2_escape(
        seed_winners=args.seed_winners,
        top_n=int(args.top_n),
        runs=int(args.runs),
        population=int(args.population),
        generations=int(args.generations),
        fitness=args.fitness,
        target=args.target,
        proposal_mode=args.proposal_mode,
        entry_rate=float(args.entry_rate),
        table_rate=float(args.table_rate),
        seed_jitter_entry_rate=float(args.seed_jitter_entry_rate),
        seed_jitter_table_rate=float(args.seed_jitter_table_rate),
        initial_mode=args.initial_mode,
        n_random_initial=int(args.n_random_initial),
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    if "q" in df and len(df):
        actual_q = int(df.q.dropna().iloc[0])
        if int(args.q) != actual_q:
            print(f"warning: CLI q={args.q} but seed winners have q={actual_q}", flush=True)
    summary = analyze_c2_escape(df)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        plot_c2_escape(df, args.plot)
    if args.save_winners:
        save_winners(args.save_winners, winners, top_n=int(args.save_winner_top_n))
    print(json.dumps(summary, indent=2))
    if args.out:
        print(f"wrote {args.out}")
        print(f"wrote {os.path.splitext(args.out)[0]}_summary.json")
    if args.plot:
        print(f"wrote {args.plot}")
    if args.save_winners:
        print(f"wrote {args.save_winners}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
