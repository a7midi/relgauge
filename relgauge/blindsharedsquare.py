"""
blindsharedsquare.py -- blind selection on the fully shared-corner square.

This is the decisive selection version of sharedsquareholonomy.py.  The
structured shared-square experiments show that copy/block/canalizing sectors
produce flat live transport, permutive sectors produce nontrivial finite
holonomy, and random sectors produce no valid transported square.  But those
experiments characterize rule ensembles that already contain structure.

This module starts from random rule tables on a single shared-corner square and
uses evolutionary consistency selection to ask:

    Does consistency discover a valid live transported square from random rules?
    If it does, is the post-hoc holonomy flat or nontrivial?

The fitness does NOT select for a group name (S_q, Z_q, etc.) and does NOT
select for any particular holonomy class unless --fitness holonomy is requested.
The default --fitness two_stage uses a smooth pretraining score first, then a
transport score asking only for all four edges to become live/bijective.
Holonomy and group structure are reported post hoc.

Recommended run
---------------
python -m relgauge.blindsharedsquare 4 ^
  --ws 1 ^
  --null-samples 200 ^
  --runs 5 ^
  --population 24 ^
  --generations 80 ^
  --fitness two_stage ^
  --pretrain-generations 30 ^
  --entry-rate 0.08 ^
  --table-rate 0.10 ^
  --initial-mode joint_random ^
  --n-random-initial 4096 ^
  --out example_results/blind_shared_square_q4_w1.csv ^
  --plot example_results/fig_blind_shared_square_q4_w1.png
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
from dataclasses import replace
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import core as C
from . import sharedsquareholonomy as SH

FitnessMode = Literal["pretrain", "transport", "two_stage", "holonomy"]
TargetMode = Literal["interface", "all"]
ProposalMode = Literal["mutate", "mixed", "random"]


# --------------------------------------------------------------------------- #
# Candidate cloning / mutation
# --------------------------------------------------------------------------- #
def clone_square(sq: SH.SharedSquare) -> SH.SharedSquare:
    """Deep-ish copy of a SharedSquare candidate, including rule tables."""
    sys = sq.joint
    new_sys = C.RelationalSystem(
        int(sys.k),
        int(sys.q),
        [tuple(int(x) for x in p) for p in sys.preds],
        [np.asarray(r, dtype=np.int64).copy() for r in sys.rules],
        meta=dict(sys.meta),
    )
    return replace(sq, joint=new_sys, meta=dict(sq.meta))


def _non_source_vertices(sq: SH.SharedSquare, target: TargetMode = "interface") -> list[int]:
    # In the shared-corner square, every non-source rule participates in the
    # interface/transport construction.  The option is left here for future
    # extension; currently target="interface" and target="all" coincide except
    # that both exclude source vertices with empty predecessor lists.
    return [v for v in range(sq.joint.k) if len(sq.joint.preds[v]) > 0]


def mutate_square(
    sq: SH.SharedSquare,
    rng: np.random.Generator,
    entry_rate: float = 0.08,
    table_rate: float = 0.10,
    target: TargetMode = "interface",
    force_one: bool = True,
) -> SH.SharedSquare:
    """Return a mutated copy of a square.

    entry_rate: probability that a local rule table is touched.
    table_rate: probability that an entry within a touched table is changed.
    """
    out = clone_square(sq)
    q = int(out.q)
    vertices = _non_source_vertices(out, target)
    touched = 0
    for v in vertices:
        if rng.random() < float(entry_rate):
            out.joint.rules[v] = SH._mutate_rule(out.joint.rules[v], q, float(table_rate), rng)
            touched += 1
    if force_one and vertices and touched == 0:
        v = int(rng.choice(vertices))
        out.joint.rules[v] = SH._mutate_rule(out.joint.rules[v], q, max(float(table_rate), 1.0 / max(1, out.joint.rules[v].size)), rng)
    out.meta = dict(out.meta)
    out.meta["blind_mutated"] = True
    return out


def random_square(q: int, w: int, rng: np.random.Generator) -> SH.SharedSquare:
    return SH.make_shared_square(int(q), int(w), rng=rng, ensemble="random", mutation_rate=0.0)


# --------------------------------------------------------------------------- #
# Evaluation / fitness
# --------------------------------------------------------------------------- #
def _min_residual_from_row(row: dict, q: int) -> float:
    keys = ["exact_B_classes", "exact_C_classes", "exact_Dtop_classes", "exact_Dbottom_classes"]
    vals = [int(row.get(k, 0) or 0) for k in keys]
    m = min(vals) if vals else 0
    return float(math.log(m, q)) if m > 0 else 0.0


def _quality_score(row: dict, q: int, w: int) -> float:
    """Smooth score used during pretraining.

    This intentionally does not select a group or holonomy.  It only rewards
    the ingredients of a live square: nontrivial edge labels, source liveness,
    edge transport, and deterministic edge maps.
    """
    min_res = float(row.get("min_residual_qcoords", 0.0) or 0.0)
    # Normalize by maximum possible residual w so w=2 does not dominate purely
    # by capacity.
    res_norm = min(1.0, max(0.0, min_res / max(1e-12, float(w))))
    src = max(0.0, min(1.0, float(row.get("source_liveness", 0.0) or 0.0)))
    tr = max(0.0, min(1.0, float(row.get("min_edge_transport", 0.0) or 0.0)))
    acc = max(0.0, min(1.0, float(row.get("min_edge_accuracy", 0.0) or 0.0)))
    # Multiplicative part prevents purely formal labels from scoring too high;
    # additive part keeps a weak gradient when one term is zero.
    product = res_norm * src * tr * acc
    additive = 0.35 * res_norm + 0.25 * src + 0.25 * tr + 0.15 * acc
    return float(0.55 * product + 0.45 * additive)


def _transport_score(row: dict, q: int, w: int) -> float:
    """Primary post-pretraining fitness: valid live/bijective transport.

    No reward is given for nontrivial holonomy here.  The aim is to test whether
    consistency selection discovers a transported gauge field at all.  Holonomy
    is measured post hoc.
    """
    min_res = float(row.get("min_residual_qcoords", 0.0) or 0.0)
    if min_res <= 0:
        # Keep a small source/transport gradient, but exact label existence is
        # the main gate.
        return float(1e-3 * (float(row.get("source_liveness", 0.0) or 0.0) + float(row.get("min_edge_transport", 0.0) or 0.0)))
    src = max(0.0, min(1.0, float(row.get("source_liveness", 0.0) or 0.0)))
    tr = max(0.0, min(1.0, float(row.get("min_edge_transport", 0.0) or 0.0)))
    acc = max(0.0, min(1.0, float(row.get("min_edge_accuracy", 0.0) or 0.0)))
    valid_bonus = 0.25 if bool(row.get("valid_square", False)) else 0.0
    return float(min_res + 0.20 * src + 0.20 * tr + 0.10 * acc + valid_bonus)


def _holonomy_score(row: dict, q: int, w: int) -> float:
    """Optional curvature-seeking score.  Not default.

    Use only after the transport-only search has been characterized.  This mode
    explicitly rewards non-flatness, so it should not be used as the primary
    selection test if the goal is to see what consistency discovers unaided.
    """
    base = _transport_score(row, q, w)
    if bool(row.get("valid_square", False)):
        base += 0.5 if bool(row.get("nontrivial_path_holonomy", False)) else 0.0
        # Reward richer generated groups very lightly.
        base += min(0.25, math.log(max(1, int(row.get("generated_group_order", 1) or 1)), 24) * 0.25)
    return float(base)


def evaluate_candidate(
    sq: SH.SharedSquare,
    rng: np.random.Generator,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    fitness: FitnessMode = "transport",
    generation: int = 0,
    pretrain_generations: int = 0,
) -> dict:
    row = SH.measure_shared_square(
        sq,
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        rng=rng,
    )
    q = int(sq.q); w = int(sq.w)
    min_res = _min_residual_from_row(row, q)
    row["min_residual_qcoords"] = float(min_res)
    if fitness == "pretrain":
        stage = "pretrain"; fit = _quality_score(row, q, w)
    elif fitness == "transport":
        stage = "transport"; fit = _transport_score(row, q, w)
    elif fitness == "holonomy":
        stage = "holonomy"; fit = _holonomy_score(row, q, w)
    elif fitness == "two_stage":
        if int(generation) < int(pretrain_generations):
            stage = "pretrain"; fit = _quality_score(row, q, w)
        else:
            stage = "transport"; fit = _transport_score(row, q, w)
    else:
        raise ValueError(f"unknown fitness {fitness!r}")
    row.update(blind_fitness=float(fit), blind_fitness_stage=stage)
    return row


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def run_null_sample(
    q: int,
    w: int,
    n_samples: int,
    initial_mode: str,
    n_random_initial: int,
    fitness: FitnessMode,
    base_seed: int = 0,
) -> tuple[pd.DataFrame, list[tuple[float, SH.SharedSquare, dict]]]:
    rows: list[dict] = []
    scored: list[tuple[float, SH.SharedSquare, dict]] = []
    for i in range(int(n_samples)):
        seed = (int(base_seed) * 1000003 + int(q) * 10007 + int(w) * 1009 + i) % 2**32
        rng = np.random.default_rng(seed)
        sq = random_square(q, w, rng)
        m = evaluate_candidate(sq, rng, initial_mode, n_random_initial, fitness="transport" if fitness == "two_stage" else fitness)
        m.update(search_mode="null", run=-1, generation=0, candidate=i, seed=int(seed), target_mode="random")
        rows.append(m); scored.append((float(m["blind_fitness"]), sq, m))
    return pd.DataFrame(rows), scored


def _select_parent(scored_pop: list[tuple[float, SH.SharedSquare, dict]], rng: np.random.Generator) -> SH.SharedSquare:
    # Tournament among the better half.
    if not scored_pop:
        raise ValueError("empty population")
    ranked = sorted(scored_pop, key=lambda t: t[0], reverse=True)
    pool = ranked[:max(1, len(ranked)//2)]
    cand = rng.choice(len(pool), size=min(3, len(pool)), replace=False)
    best = max((pool[int(i)] for i in np.atleast_1d(cand)), key=lambda t: t[0])
    return best[1]


def run_evolution(
    q: int,
    w: int,
    runs: int,
    population: int,
    generations: int,
    fitness: FitnessMode,
    pretrain_generations: int,
    target: TargetMode,
    proposal_mode: ProposalMode,
    entry_rate: float,
    table_rate: float,
    initial_mode: str,
    n_random_initial: int,
    base_seed: int = 12345,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[float, SH.SharedSquare, dict]]]:
    rows: list[dict] = []
    winner_pool: list[tuple[float, SH.SharedSquare, dict]] = []
    for run in range(int(runs)):
        rng = np.random.default_rng((int(base_seed) + 7919 * run + 17 * q + w) % 2**32)
        pop = [random_square(q, w, rng) for _ in range(int(population))]
        scored: list[tuple[float, SH.SharedSquare, dict]] = []
        for gen in range(int(generations) + 1):
            scored = []
            for ci, sq in enumerate(pop):
                erng = np.random.default_rng(rng.integers(0, 2**32 - 1, dtype=np.uint32).item())
                m = evaluate_candidate(sq, erng, initial_mode, n_random_initial, fitness=fitness, generation=gen, pretrain_generations=pretrain_generations)
                m.update(search_mode="evolve", run=int(run), generation=int(gen), candidate=int(ci), seed=np.nan, target_mode=target, proposal_mode=proposal_mode, entry_rate=float(entry_rate), table_rate=float(table_rate), fitness_mode=fitness)
                rows.append(m); scored.append((float(m["blind_fitness"]), sq, m))
            scored.sort(key=lambda t: t[0], reverse=True)
            best = scored[0][2]
            winner_pool.extend(scored[:min(3, len(scored))])
            if verbose:
                print(
                    "blind-shared-square "
                    f"run={run} gen={gen} stage={best.get('blind_fitness_stage')} "
                    f"best={best.get('blind_fitness'):.4f} minres={best.get('min_residual_qcoords'):.4f} "
                    f"src={best.get('source_liveness'):.4f} trans={best.get('min_edge_transport'):.4f} "
                    f"valid={bool(best.get('valid_square'))} hol={bool(best.get('nontrivial_path_holonomy'))}",
                    flush=True,
                )
            if gen == int(generations):
                break
            # Elitism plus mutations.
            n_elite = max(2, int(population)//6)
            next_pop = [clone_square(t[1]) for t in scored[:n_elite]]
            while len(next_pop) < int(population):
                if proposal_mode == "random":
                    child = random_square(q, w, rng)
                elif proposal_mode == "mixed" and rng.random() < 0.10:
                    child = random_square(q, w, rng)
                else:
                    parent = _select_parent(scored, rng)
                    child = mutate_square(parent, rng, entry_rate=entry_rate, table_rate=table_rate, target=target)
                next_pop.append(child)
            pop = next_pop
    return pd.DataFrame(rows), winner_pool


def run_blind_shared_square(
    q: int = 4,
    ws: Iterable[int] = (1,),
    null_samples: int = 200,
    runs: int = 5,
    population: int = 24,
    generations: int = 80,
    fitness: FitnessMode = "two_stage",
    pretrain_generations: int = 30,
    target: TargetMode = "interface",
    proposal_mode: ProposalMode = "mixed",
    entry_rate: float = 0.08,
    table_rate: float = 0.10,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[float, SH.SharedSquare, dict]]]:
    all_rows: list[pd.DataFrame] = []
    all_winners: list[tuple[float, SH.SharedSquare, dict]] = []
    for w in ws:
        nd, nw = run_null_sample(q, int(w), null_samples, initial_mode, n_random_initial, fitness, base_seed=base_seed + 31 * int(w))
        all_rows.append(nd); all_winners.extend(nw)
        if verbose:
            print(f"blind-shared-square null q={q} w={w} done", flush=True)
        ed, ew = run_evolution(q, int(w), runs, population, generations, fitness, pretrain_generations, target, proposal_mode, entry_rate, table_rate, initial_mode, n_random_initial, base_seed=base_seed + 1009 * int(w), verbose=verbose)
        all_rows.append(ed); all_winners.extend(ew)
        if verbose:
            print(f"blind-shared-square evolve q={q} w={w} done", flush=True)
    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return df, all_winners


# --------------------------------------------------------------------------- #
# Analysis / plotting / winner saving
# --------------------------------------------------------------------------- #
def _best_by_run_generation(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df.search_mode == "evolve"].copy()
    if ev.empty:
        return ev
    idx = ev.groupby(["w", "run", "generation"])["blind_fitness"].idxmax()
    return ev.loc[idx].copy()


def analyze_blind_shared_square(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    null = df[df.search_mode == "null"].copy()
    best = _best_by_run_generation(df)
    final = best.loc[best.groupby(["w", "run"])["generation"].idxmax()] if len(best) else best
    all_time = df.loc[df["blind_fitness"].idxmax()].to_dict() if len(df) else {}
    null_valid = float(null.valid_square.mean()) if len(null) else math.nan
    null_hol = float(null[null.valid_square.astype(bool)].nontrivial_path_holonomy.mean()) if len(null[null.valid_square.astype(bool)]) else 0.0
    final_valid = float(final.valid_square.mean()) if len(final) else math.nan
    final_hol = float(final[final.valid_square.astype(bool)].nontrivial_path_holonomy.mean()) if len(final[final.valid_square.astype(bool)]) else 0.0
    best_valid = bool(all_time.get("valid_square", False))
    best_hol = bool(all_time.get("nontrivial_path_holonomy", False))
    if final_valid > 0 and final_hol > 0:
        verdict = "BLIND SHARED-SQUARE HOLONOMY SIGNAL: random-start selection finds valid nontrivial gauge-covariant squares"
    elif final_valid > 0:
        verdict = "BLIND SHARED-SQUARE TRANSPORT SIGNAL: random-start selection finds valid but mostly flat squares"
    elif best_valid:
        verdict = "PARTIAL BLIND SHARED-SQUARE SIGNAL: at least one valid square found, not robust across runs"
    else:
        verdict = "NO BLIND SHARED-SQUARE SIGNAL in this regime"
    by_gen = []
    if len(best):
        for gen, g in best.groupby("generation"):
            gv = g[g.valid_square.astype(bool)]
            by_gen.append(dict(
                generation=int(gen),
                best_fitness=float(g.blind_fitness.max()),
                mean_best_fitness=float(g.blind_fitness.mean()),
                valid_run_fraction=float(g.valid_square.mean()),
                holonomy_run_fraction=float(gv.nontrivial_path_holonomy.mean()) if len(gv) else 0.0,
                best_min_residual=float(g.min_residual_qcoords.max()),
                best_source=float(g.source_liveness.max()),
                best_transport=float(g.min_edge_transport.max()),
                stage=str(g.blind_fitness_stage.iloc[0]) if "blind_fitness_stage" in g else "",
            ))
    valid = df[df.valid_square.astype(bool)] if "valid_square" in df else df.iloc[0:0]
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        null_valid_square_fraction=null_valid,
        null_nontrivial_holonomy_fraction=null_hol,
        null_best_fitness=float(null.blind_fitness.max()) if len(null) else math.nan,
        final_valid_run_fraction=final_valid,
        final_nontrivial_holonomy_run_fraction=final_hol,
        evolved_final_best_fitness=float(final.blind_fitness.mean()) if len(final) else math.nan,
        all_time_best_fitness=float(all_time.get("blind_fitness", math.nan)),
        all_time_best_valid_square=best_valid,
        all_time_best_nontrivial_holonomy=best_hol,
        all_time_best_min_residual=float(all_time.get("min_residual_qcoords", math.nan)),
        all_time_best_source_liveness=float(all_time.get("source_liveness", math.nan)),
        all_time_best_min_edge_transport=float(all_time.get("min_edge_transport", math.nan)),
        valid_square_fraction=float(df.valid_square.mean()) if "valid_square" in df else math.nan,
        nontrivial_holonomy_fraction=float(valid.nontrivial_path_holonomy.mean()) if len(valid) else 0.0,
        gauge_covariance_success=float(valid.gauge_covariance_success.dropna().mean()) if len(valid) else math.nan,
        conjugacy_class_success=float(valid.conjugacy_class_success.dropna().mean()) if len(valid) else math.nan,
        generated_group_counts={str(k): int(v) for k, v in (valid.generated_group_name.value_counts().items() if len(valid) else [])},
        delta_type_counts={str(k): int(v) for k, v in (valid.delta_type.value_counts().items() if len(valid) else [])},
        best_by_generation=by_gen,
    )


def save_winners(path: str, winners: list[tuple[float, SH.SharedSquare, dict]], top_n: int = 50, min_valid: bool = False) -> None:
    ranked = sorted(winners, key=lambda t: (
        bool(t[2].get("valid_square", False)),
        bool(t[2].get("nontrivial_path_holonomy", False)),
        float(t[2].get("blind_fitness", t[0])),
        float(t[2].get("min_residual_qcoords", 0.0)),
    ), reverse=True)
    out = []
    for fit, sq, row in ranked:
        if min_valid and not bool(row.get("valid_square", False)):
            continue
        out.append(dict(fitness=float(fit), metrics=dict(row), square=sq))
        if len(out) >= int(top_n):
            break
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(kind="blind_shared_square_winners", winners=out), f)


def plot_blind_shared_square(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    best = _best_by_run_generation(df)
    if len(best):
        g = best.groupby("generation")
        ax[0].plot(list(g.groups), [g.get_group(k).blind_fitness.max() for k in g.groups], "o-", label="global best")
        ax[0].plot(list(g.groups), [g.get_group(k).blind_fitness.mean() for k in g.groups], "o-", label="mean run-best")
        ax[0].set_title("blind shared-square fitness")
        ax[0].set_xlabel("generation"); ax[0].set_ylabel("fitness")
        ax[0].legend(fontsize=8)
        ax[1].plot(list(g.groups), [g.get_group(k).valid_square.mean() for k in g.groups], "o-", label="valid")
        ax[1].plot(list(g.groups), [g.get_group(k)[g.get_group(k).valid_square.astype(bool)].nontrivial_path_holonomy.mean() if len(g.get_group(k)[g.get_group(k).valid_square.astype(bool)]) else 0.0 for k in g.groups], "o-", label="nontriv holonomy")
        ax[1].set_title("valid transport / holonomy over time")
        ax[1].set_xlabel("generation"); ax[1].set_ylabel("fraction among run-bests")
        ax[1].legend(fontsize=8)
    ev = df[df.search_mode == "evolve"]
    if len(ev):
        ax[2].scatter(ev.source_liveness, ev.min_edge_transport, s=np.maximum(5, ev.blind_fitness * 20), alpha=0.45, c=ev.valid_square.astype(int))
        ax[2].set_title("source-liveness vs edge transport")
        ax[2].set_xlabel("source liveness")
        ax[2].set_ylabel("min edge transport")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_list(s: str, typ=int):
    if not s:
        return []
    return [typ(x) for x in str(s).split(",") if str(x).strip()]


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Blind evolutionary selection on the fully shared-corner square.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--ws", default="1")
    p.add_argument("--null-samples", type=int, default=200)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--generations", type=int, default=80)
    p.add_argument("--fitness", choices=["pretrain", "transport", "two_stage", "holonomy"], default="two_stage")
    p.add_argument("--pretrain-generations", type=int, default=30)
    p.add_argument("--target", choices=["interface", "all"], default="interface")
    p.add_argument("--proposal-mode", choices=["mutate", "mixed", "random"], default="mixed")
    p.add_argument("--entry-rate", type=float, default=0.08)
    p.add_argument("--table-rate", type=float, default=0.10)
    p.add_argument("--initial-mode", choices=["source_all", "source_random", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--save-winners", default=None)
    p.add_argument("--save-winner-top-n", type=int, default=50)
    p.add_argument("--save-only-valid", action="store_true")
    p.add_argument("--out", default="example_results/blind_shared_square.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    df, winners = run_blind_shared_square(
        q=int(args.q),
        ws=_parse_list(args.ws, int),
        null_samples=int(args.null_samples),
        runs=int(args.runs),
        population=int(args.population),
        generations=int(args.generations),
        fitness=args.fitness,
        pretrain_generations=int(args.pretrain_generations),
        target=args.target,
        proposal_mode=args.proposal_mode,
        entry_rate=float(args.entry_rate),
        table_rate=float(args.table_rate),
        initial_mode=args.initial_mode,
        n_random_initial=int(args.n_random_initial),
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    summary = analyze_blind_shared_square(df)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.save_winners:
        save_winners(args.save_winners, winners, top_n=int(args.save_winner_top_n), min_valid=bool(args.save_only_valid))
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_blind_shared_square(df, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
