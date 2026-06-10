"""
blindmicrolattice.py -- blind selection on a coupled microscopic plaquette lattice.

This is the selection version of ``microlattice.py``.  The structured
microscopic lattice controls show that a hand-built ``c2link`` sector gives a
valid shared-edge Z2 lattice, while random rules almost never produce valid
transported plaquettes.  This module asks the selection question:

    Starting from random rule tables on a coupled lattice whose plaquettes
    literally share edge transport systems, can consistency selection discover
    a coherent Z2/C2 transported plaquette lattice?

The default fitness rewards only the ingredients of a valid transported
microscopic lattice: nontrivial shared labels on plaquettes, high edge-transport
between neighbouring labels, deterministic edge maps, and a high fraction of
valid C2 plaquettes.  It does NOT reward a target Wilson-loop law, a target
flux density, or nontrivial curvature.  Flux statistics and Wilson-loop
observables are reported post hoc.

Recommended run
---------------
python -m relgauge.blindmicrolattice 4 ^
  --sizes 3x3 ^
  --null-samples 100 ^
  --runs 5 ^
  --population 24 ^
  --generations 100 ^
  --fitness two_stage ^
  --pretrain-generations 40 ^
  --entry-rate 0.06 ^
  --table-rate 0.08 ^
  --initial-mode joint_random ^
  --n-random-initial 4096 ^
  --out example_results/blind_microlattice_q4_3x3.csv ^
  --plot example_results/fig_blind_microlattice_q4_3x3.png

For a faster first check use --sizes 2x2, --population 12, --generations 30,
and --n-random-initial 1024.
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
from . import microlattice as ML
from . import sharedsquareholonomy as SH

FitnessMode = Literal["pretrain", "transport", "two_stage", "flux"]
ProposalMode = Literal["mutate", "mixed", "random"]
TargetMode = Literal["interface", "all"]


# --------------------------------------------------------------------------- #
# Candidate creation / mutation
# --------------------------------------------------------------------------- #
def clone_lattice(lat: ML.MicroLattice) -> ML.MicroLattice:
    """Copy a MicroLattice candidate, preserving topology and metadata."""
    sys = lat.joint
    new_sys = C.RelationalSystem(
        int(sys.k), int(sys.q),
        [tuple(int(x) for x in p) for p in sys.preds],
        [np.asarray(r, dtype=np.int64).copy() for r in sys.rules],
        meta=dict(sys.meta),
    )
    # EdgeData objects only contain topology and flip metadata; they are not
    # mutated, so a shallow copy of the dict is enough.
    return replace(lat, joint=new_sys, node_out=dict(lat.node_out), edges=dict(lat.edges), meta=dict(lat.meta))


def random_lattice(q: int, nx: int, ny: int, w: int, rng: np.random.Generator) -> ML.MicroLattice:
    return ML.make_microscopic_lattice(q=int(q), nx=int(nx), ny=int(ny), w=int(w), rng=rng, ensemble="random")


def _mutatable_vertices(lat: ML.MicroLattice, target: TargetMode = "interface") -> list[int]:
    # For the current microscopic lattice, every non-source vertex participates
    # in edge transport or corner propagation.  Keep the target option for API
    # symmetry with blindsharedsquare.
    return [v for v in range(lat.joint.k) if len(lat.joint.preds[v]) > 0]


def mutate_lattice(
    lat: ML.MicroLattice,
    rng: np.random.Generator,
    entry_rate: float = 0.06,
    table_rate: float = 0.08,
    target: TargetMode = "interface",
    force_one: bool = True,
) -> ML.MicroLattice:
    out = clone_lattice(lat)
    q = int(out.q)
    vertices = _mutatable_vertices(out, target)
    touched = 0
    for v in vertices:
        if rng.random() < float(entry_rate):
            out.joint.rules[v] = SH._mutate_rule(out.joint.rules[v], q, float(table_rate), rng)
            touched += 1
    if force_one and vertices and touched == 0:
        v = int(rng.choice(vertices))
        # Ensure at least one table entry can change.
        rate = max(float(table_rate), 1.0 / max(1, int(out.joint.rules[v].size)))
        out.joint.rules[v] = SH._mutate_rule(out.joint.rules[v], q, rate, rng)
    out.meta = dict(out.meta)
    out.meta["blind_mutated"] = True
    return out


# --------------------------------------------------------------------------- #
# Evaluation / fitness
# --------------------------------------------------------------------------- #
def _plaquette_stats(meas: dict, q: int, w: int) -> dict:
    plaquettes = list(meas.get("plaquettes", []))
    if not plaquettes:
        return dict(
            mean_residual_qcoords=0.0, min_residual_qcoords=0.0,
            mean_residual_norm=0.0, min_residual_norm=0.0,
            mean_edge_transport=0.0, min_edge_transport=0.0,
            mean_edge_accuracy=0.0, min_edge_accuracy=0.0,
        )
    residuals = []
    transports = []
    accuracies = []
    for p in plaquettes:
        nc = int(p.get("n_classes", 0) or 0)
        residuals.append(float(math.log(nc, q)) if nc > 1 else 0.0)
        transports.append(float(p.get("min_transport", 0.0) or 0.0))
        accuracies.append(float(p.get("min_accuracy", 0.0) or 0.0))
    wnorm = max(1e-12, float(w))
    return dict(
        mean_residual_qcoords=float(np.mean(residuals)),
        min_residual_qcoords=float(np.min(residuals)),
        mean_residual_norm=float(np.mean(residuals) / wnorm),
        min_residual_norm=float(np.min(residuals) / wnorm),
        mean_edge_transport=float(np.mean(transports)),
        min_edge_transport=float(np.min(transports)),
        mean_edge_accuracy=float(np.mean(accuracies)),
        min_edge_accuracy=float(np.min(accuracies)),
    )


def _wilson_summary(flux_grid: np.ndarray, max_side: int = 3) -> dict:
    vals_by_area: dict[int, list[float]] = {}
    ny, nx = flux_grid.shape
    for a in range(1, min(int(max_side), nx) + 1):
        for b in range(1, min(int(max_side), ny) + 1):
            vals = ML._rect_wilson_values(flux_grid, a, b)
            if vals:
                vals_by_area.setdefault(a*b, []).extend(vals)
    out: dict[str, float | int] = {}
    for area, vals in sorted(vals_by_area.items()):
        out[f"wilson_area_{area}_mean"] = float(np.mean(vals))
        out[f"wilson_area_{area}_abs"] = float(abs(np.mean(vals)))
        out[f"wilson_area_{area}_n"] = int(len(vals))
    return out


def _quality_score(row: dict, w: int) -> float:
    """Smooth pretraining score.

    This gives a gradient even when no plaquette is fully valid yet.  It rewards
    residual labels, transport, deterministic maps, and emerging valid C2
    plaquettes.  It does not reward flux sign or Wilson-loop behavior.
    """
    res = max(0.0, min(1.0, float(row.get("mean_residual_norm", 0.0) or 0.0)))
    tr = max(0.0, min(1.0, float(row.get("mean_edge_transport", 0.0) or 0.0)))
    acc = max(0.0, min(1.0, float(row.get("mean_edge_accuracy", 0.0) or 0.0)))
    valid = max(0.0, min(1.0, float(row.get("valid_plaquette_fraction", 0.0) or 0.0)))
    z2 = max(0.0, min(1.0, float(row.get("z2_plaquette_fraction", 0.0) or 0.0)))
    product = res * tr * acc
    additive = 0.30*res + 0.25*tr + 0.20*acc + 0.15*valid + 0.10*z2
    return float(0.55*product + 0.45*additive)


def _transport_score(row: dict, w: int) -> float:
    """Primary selection score: coherent valid C2 transported plaquettes.

    Larger residual is rewarded, but the main gate is the fraction of plaquettes
    that are valid C2 plaquettes.  No direct reward is given for nontrivial
    curvature or any Wilson-loop law.
    """
    z2 = max(0.0, min(1.0, float(row.get("z2_plaquette_fraction", 0.0) or 0.0)))
    valid = max(0.0, min(1.0, float(row.get("valid_plaquette_fraction", 0.0) or 0.0)))
    res = max(0.0, min(1.0, float(row.get("mean_residual_norm", 0.0) or 0.0)))
    minres = max(0.0, min(1.0, float(row.get("min_residual_norm", 0.0) or 0.0)))
    tr = max(0.0, min(1.0, float(row.get("mean_edge_transport", 0.0) or 0.0)))
    mintr = max(0.0, min(1.0, float(row.get("min_edge_transport", 0.0) or 0.0)))
    acc = max(0.0, min(1.0, float(row.get("mean_edge_accuracy", 0.0) or 0.0)))
    return float(1.35*z2 + 0.45*valid + 0.25*res + 0.15*minres + 0.20*tr + 0.10*mintr + 0.10*acc)


def _flux_score(row: dict, w: int) -> float:
    """Optional curvature/flux-seeking score; not the default."""
    base = _transport_score(row, w)
    non = max(0.0, min(1.0, float(row.get("nontrivial_plaquette_fraction", 0.0) or 0.0)))
    # Reward a mix of identity and nontrivial plaquettes weakly.
    mix = 4.0 * non * (1.0 - non)
    return float(base + 0.30*non + 0.20*mix)


def evaluate_candidate(
    lat: ML.MicroLattice,
    rng: np.random.Generator,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    fitness: FitnessMode = "transport",
    generation: int = 0,
    pretrain_generations: int = 0,
    max_loop_side: int = 3,
) -> dict:
    meas = ML.measure_microscopic_lattice(
        lat,
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        rng=rng,
    )
    row: dict = dict(
        q=int(lat.q), w=int(lat.w), nx=int(lat.nx), ny=int(lat.ny),
        valid_plaquette_fraction=float(meas["valid_fraction"]),
        z2_plaquette_fraction=float(meas["z2_fraction"]),
        nontrivial_plaquette_fraction=float(meas["nontrivial_fraction"]),
        delta_type_counts=json.dumps(meas.get("delta_counts", {})),
        group_name_counts=json.dumps(meas.get("group_counts", {})),
    )
    row.update(_plaquette_stats(meas, int(lat.q), int(lat.w)))
    row.update(_wilson_summary(meas["flux_grid"], max_side=int(max_loop_side)))
    if fitness == "pretrain":
        stage = "pretrain"; fit = _quality_score(row, int(lat.w))
    elif fitness == "transport":
        stage = "transport"; fit = _transport_score(row, int(lat.w))
    elif fitness == "flux":
        stage = "flux"; fit = _flux_score(row, int(lat.w))
    elif fitness == "two_stage":
        if int(generation) < int(pretrain_generations):
            stage = "pretrain"; fit = _quality_score(row, int(lat.w))
        else:
            stage = "transport"; fit = _transport_score(row, int(lat.w))
    else:
        raise ValueError(f"unknown fitness {fitness!r}")
    row.update(blind_fitness=float(fit), blind_fitness_stage=stage)
    # Convenient booleans for summary.
    row["selected_lattice"] = bool(row["z2_plaquette_fraction"] >= 0.80 and row["mean_edge_transport"] >= 0.80)
    row["full_valid_lattice"] = bool(row["z2_plaquette_fraction"] >= 0.999)
    row["nontrivial_flux"] = bool(row["nontrivial_plaquette_fraction"] > 0.05)
    return row


# --------------------------------------------------------------------------- #
# Evolutionary search
# --------------------------------------------------------------------------- #
def _select_parent(scored_pop: list[tuple[float, ML.MicroLattice, dict]], rng: np.random.Generator) -> ML.MicroLattice:
    ordered = sorted(scored_pop, key=lambda t: t[0], reverse=True)
    pool = ordered[:max(1, len(ordered)//2)]
    contestants = rng.choice(len(pool), size=min(3, len(pool)), replace=False)
    best = max((pool[int(i)] for i in contestants), key=lambda t: t[0])
    return best[1]


def _best_by_run_generation(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df.search_mode == "evolve"].copy() if "search_mode" in df else df.iloc[0:0].copy()
    if ev.empty:
        return ev
    idx = ev.groupby(["nx", "ny", "run", "generation"])["blind_fitness"].idxmax()
    return ev.loc[idx].copy()


def run_blind_microlattice(
    q: int = 4,
    sizes: Iterable[tuple[int, int]] = ((3, 3),),
    w: int = 1,
    null_samples: int = 100,
    runs: int = 5,
    population: int = 24,
    generations: int = 100,
    fitness: FitnessMode = "two_stage",
    pretrain_generations: int = 40,
    proposal_mode: ProposalMode = "mixed",
    target: TargetMode = "interface",
    entry_rate: float = 0.06,
    table_rate: float = 0.08,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    max_loop_side: int = 3,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[float, ML.MicroLattice, dict]]]:
    rows: list[dict] = []
    winners: list[tuple[float, ML.MicroLattice, dict]] = []
    for nx, ny in sizes:
        # Null sample: random lattices evaluated by the final transport score.
        for i in range(int(null_samples)):
            seed = (int(base_seed)*1000003 + int(q)*10007 + int(nx)*1009 + int(ny)*917 + i) % 2**32
            rng = np.random.default_rng(seed)
            lat = random_lattice(q, nx, ny, w, rng)
            m = evaluate_candidate(lat, rng, initial_mode, n_random_initial, fitness="transport" if fitness == "two_stage" else fitness, max_loop_side=max_loop_side)
            m.update(search_mode="null", run=-1, generation=0, candidate=i, seed=int(seed), proposal_mode="random")
            rows.append(m)
        if verbose:
            print(f"blind-microlattice null q={q} size={nx}x{ny} done", flush=True)
        for run in range(int(runs)):
            seed0 = (int(base_seed)*2000003 + int(q)*20011 + int(nx)*2017 + int(ny)*2027 + int(run)*97) % 2**32
            rng = np.random.default_rng(seed0)
            pop = [random_lattice(q, nx, ny, w, rng) for _ in range(int(population))]
            scored: list[tuple[float, ML.MicroLattice, dict]] = []
            for gen in range(int(generations) + 1):
                scored = []
                for ci, cand in enumerate(pop):
                    m = evaluate_candidate(cand, rng, initial_mode, n_random_initial, fitness=fitness, generation=gen, pretrain_generations=pretrain_generations, max_loop_side=max_loop_side)
                    m.update(search_mode="evolve", run=int(run), generation=int(gen), candidate=int(ci), seed=int(seed0), proposal_mode=str(proposal_mode), target_mode=str(target))
                    rows.append(m)
                    scored.append((float(m["blind_fitness"]), cand, m))
                    winners.append((float(m["blind_fitness"]), cand, m))
                best = max(scored, key=lambda t: t[0])
                if verbose:
                    bm = best[2]
                    print(
                        f"blind-microlattice size={nx}x{ny} run={run} gen={gen} "
                        f"stage={bm.get('blind_fitness_stage')} best={best[0]:.4f} "
                        f"z2={float(bm.get('z2_plaquette_fraction',0)):.3f} "
                        f"valid={float(bm.get('valid_plaquette_fraction',0)):.3f} "
                        f"trans={float(bm.get('mean_edge_transport',0)):.3f} "
                        f"nontriv={float(bm.get('nontrivial_plaquette_fraction',0)):.3f}",
                        flush=True,
                    )
                if gen == int(generations):
                    break
                # Elitism plus mutation/restarts.
                ordered = sorted(scored, key=lambda t: t[0], reverse=True)
                elite_n = max(1, int(round(0.15 * int(population))))
                newpop = [clone_lattice(t[1]) for t in ordered[:elite_n]]
                while len(newpop) < int(population):
                    if proposal_mode in ("mixed", "random") and rng.random() < (0.10 if proposal_mode == "mixed" else 1.0):
                        child = random_lattice(q, nx, ny, w, rng)
                    else:
                        parent = _select_parent(scored, rng)
                        child = mutate_lattice(parent, rng, entry_rate=entry_rate, table_rate=table_rate, target=target)
                    newpop.append(child)
                pop = newpop
    return pd.DataFrame(rows), winners


# --------------------------------------------------------------------------- #
# Analysis / plotting / saving
# --------------------------------------------------------------------------- #
def analyze_blind_microlattice(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    null = df[df.search_mode == "null"].copy()
    best = _best_by_run_generation(df)
    final = best.loc[best.groupby(["nx", "ny", "run"])["generation"].idxmax()] if len(best) else best
    all_time = df.loc[df["blind_fitness"].idxmax()].to_dict() if len(df) else {}
    null_selected = float(null.selected_lattice.mean()) if len(null) else math.nan
    final_selected = float(final.selected_lattice.mean()) if len(final) else math.nan
    final_full = float(final.full_valid_lattice.mean()) if len(final) else math.nan
    final_nontriv = float(final[final.selected_lattice.astype(bool)].nontrivial_flux.mean()) if len(final[final.selected_lattice.astype(bool)]) else 0.0
    if final_selected > 0 and final_nontriv > 0:
        verdict = "BLIND MICRO-LATTICE Z2 FLUX SIGNAL: random-start selection finds coupled valid C2 plaquette lattices with nontrivial flux"
    elif final_selected > 0:
        verdict = "BLIND MICRO-LATTICE TRANSPORT SIGNAL: random-start selection finds coupled valid C2 plaquette lattices, mostly flat"
    elif bool(all_time.get("selected_lattice", False)):
        verdict = "PARTIAL BLIND MICRO-LATTICE SIGNAL: at least one selected lattice found, not robust across runs"
    else:
        verdict = "NO BLIND MICRO-LATTICE SIGNAL in this regime"
    by_gen = []
    if len(best):
        for gen, g in best.groupby("generation"):
            by_gen.append(dict(
                generation=int(gen),
                best_fitness=float(g.blind_fitness.max()),
                mean_best_fitness=float(g.blind_fitness.mean()),
                selected_run_fraction=float(g.selected_lattice.mean()),
                full_valid_run_fraction=float(g.full_valid_lattice.mean()),
                mean_z2_fraction=float(g.z2_plaquette_fraction.mean()),
                best_z2_fraction=float(g.z2_plaquette_fraction.max()),
                mean_nontrivial_fraction=float(g.nontrivial_plaquette_fraction.mean()),
                best_nontrivial_fraction=float(g.nontrivial_plaquette_fraction.max()),
                best_mean_transport=float(g.mean_edge_transport.max()),
                stage=str(g.blind_fitness_stage.iloc[0]) if "blind_fitness_stage" in g else "",
            ))
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        null_selected_lattice_fraction=null_selected,
        null_best_fitness=float(null.blind_fitness.max()) if len(null) else math.nan,
        null_best_z2_fraction=float(null.z2_plaquette_fraction.max()) if len(null) else math.nan,
        evolved_final_selected_lattice_fraction=final_selected,
        evolved_final_full_valid_fraction=final_full,
        evolved_final_nontrivial_flux_fraction=final_nontriv,
        evolved_final_best_fitness=float(final.blind_fitness.mean()) if len(final) else math.nan,
        all_time_best_fitness=float(all_time.get("blind_fitness", math.nan)),
        all_time_best_selected_lattice=bool(all_time.get("selected_lattice", False)),
        all_time_best_full_valid=bool(all_time.get("full_valid_lattice", False)),
        all_time_best_z2_fraction=float(all_time.get("z2_plaquette_fraction", math.nan)),
        all_time_best_valid_fraction=float(all_time.get("valid_plaquette_fraction", math.nan)),
        all_time_best_nontrivial_fraction=float(all_time.get("nontrivial_plaquette_fraction", math.nan)),
        all_time_best_mean_transport=float(all_time.get("mean_edge_transport", math.nan)),
        final_group_name_counts=_merged_json_counts(final.get("group_name_counts", pd.Series(dtype=str))) if len(final) else {},
        final_delta_type_counts=_merged_json_counts(final.get("delta_type_counts", pd.Series(dtype=str))) if len(final) else {},
        best_by_generation=by_gen,
    )


def _merged_json_counts(series: pd.Series) -> dict:
    counts: dict[str, int] = {}
    for s in series.dropna().tolist():
        try:
            d = json.loads(s) if isinstance(s, str) else dict(s)
        except Exception:
            d = {}
        for k, v in d.items():
            counts[str(k)] = counts.get(str(k), 0) + int(v)
    return counts


def save_winners(path: str, winners: list[tuple[float, ML.MicroLattice, dict]], top_n: int = 50, min_selected: bool = False) -> None:
    ranked = sorted(winners, key=lambda t: (
        bool(t[2].get("selected_lattice", False)),
        bool(t[2].get("full_valid_lattice", False)),
        float(t[2].get("z2_plaquette_fraction", 0.0)),
        float(t[2].get("blind_fitness", t[0])),
        float(t[2].get("nontrivial_plaquette_fraction", 0.0)),
    ), reverse=True)
    out = []
    for fit, lat, row in ranked:
        if min_selected and not bool(row.get("selected_lattice", False)):
            continue
        out.append(dict(fitness=float(fit), metrics=dict(row), lattice=lat))
        if len(out) >= int(top_n):
            break
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(kind="blind_microlattice_winners", winners=out), f)


def plot_blind_microlattice(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    best = _best_by_run_generation(df)
    if len(best):
        g = best.groupby("generation")
        xs = list(g.groups)
        ax[0].plot(xs, [g.get_group(k).blind_fitness.max() for k in xs], "o-", label="global best")
        ax[0].plot(xs, [g.get_group(k).blind_fitness.mean() for k in xs], "o-", label="mean run-best")
        ax[0].set_title("blind micro-lattice fitness")
        ax[0].set_xlabel("generation"); ax[0].set_ylabel("fitness")
        ax[0].legend(fontsize=8)
        ax[1].plot(xs, [g.get_group(k).selected_lattice.mean() for k in xs], "o-", label="selected lattice")
        ax[1].plot(xs, [g.get_group(k).z2_plaquette_fraction.mean() for k in xs], "o-", label="mean z2 plaquettes")
        ax[1].plot(xs, [g.get_group(k).nontrivial_plaquette_fraction.mean() for k in xs], "o-", label="mean nontriv flux")
        ax[1].set_title("valid C2 plaquette lattice over time")
        ax[1].set_xlabel("generation"); ax[1].set_ylabel("fraction")
        ax[1].legend(fontsize=8)
    ev = df[df.search_mode == "evolve"]
    if len(ev):
        ax[2].scatter(ev.z2_plaquette_fraction, ev.nontrivial_plaquette_fraction, s=np.maximum(5, ev.blind_fitness * 15), alpha=0.45, c=ev.selected_lattice.astype(int))
        ax[2].set_title("C2 plaquette validity vs flux")
        ax[2].set_xlabel("z2 plaquette fraction")
        ax[2].set_ylabel("nontrivial plaquette fraction")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_sizes(s: str) -> list[tuple[int,int]]:
    out = []
    for part in str(s).split(','):
        part = part.strip().lower()
        if not part:
            continue
        if 'x' in part:
            a,b = part.split('x',1); out.append((int(a), int(b)))
        else:
            n = int(part); out.append((n,n))
    return out


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Blind selection on microscopic coupled C2 plaquette lattices.")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--sizes", default="3x3")
    p.add_argument("--w", type=int, default=1)
    p.add_argument("--null-samples", type=int, default=100)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--fitness", choices=["pretrain", "transport", "two_stage", "flux"], default="two_stage")
    p.add_argument("--pretrain-generations", type=int, default=40)
    p.add_argument("--proposal-mode", choices=["mutate", "mixed", "random"], default="mixed")
    p.add_argument("--target", choices=["interface", "all"], default="interface")
    p.add_argument("--entry-rate", type=float, default=0.06)
    p.add_argument("--table-rate", type=float, default=0.08)
    p.add_argument("--initial-mode", choices=["source_all", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--max-loop-side", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--save-winners", default=None)
    p.add_argument("--save-winner-top-n", type=int, default=50)
    p.add_argument("--save-only-selected", action="store_true")
    p.add_argument("--out", default="example_results/blind_microlattice.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    df, winners = run_blind_microlattice(
        q=int(args.q), sizes=_parse_sizes(args.sizes), w=int(args.w),
        null_samples=int(args.null_samples), runs=int(args.runs), population=int(args.population), generations=int(args.generations),
        fitness=args.fitness, pretrain_generations=int(args.pretrain_generations), proposal_mode=args.proposal_mode,
        target=args.target, entry_rate=float(args.entry_rate), table_rate=float(args.table_rate),
        initial_mode=args.initial_mode, n_random_initial=int(args.n_random_initial), max_loop_side=int(args.max_loop_side),
        base_seed=int(args.base_seed), verbose=not bool(args.quiet),
    )
    summary = analyze_blind_microlattice(df)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.save_winners:
        save_winners(args.save_winners, winners, top_n=int(args.save_winner_top_n), min_selected=bool(args.save_only_selected))
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_blind_microlattice(df, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
