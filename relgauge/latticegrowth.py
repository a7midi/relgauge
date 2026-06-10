"""
latticegrowth.py -- curriculum growth of selected microscopic Z2 lattices.

This module tests the next claim after blind microscopic-lattice selection:

    A selected 3x3 C2/Z2 plaquette lattice can be extended to a 4x4 lattice
    by adding new random edge-transport systems and applying the same
    consistency/transport selection, rather than rebuilding the whole lattice
    from scratch.

It loads saved winners from ``relgauge.blindmicrolattice``, embeds each winner
as the north-west core of a larger microscopic lattice, freezes or soft-mutates
that inherited core, and evolves only the new growth region by default.  The
fitness is the same transported-plaquette fitness used for blind micro-lattices;
no target flux map, Wilson-loop law, or hand-coded plaquette pattern is rewarded.

Recommended run
---------------
python -m relgauge.latticegrowth 4 ^
  --seed-winners example_results/blind_microlattice_winners_q4_3x3.pkl ^
  --from-size 3x3 ^
  --to-size 4x4 ^
  --top-n 20 ^
  --runs 10 ^
  --population 40 ^
  --generations 150 ^
  --fitness two_stage ^
  --pretrain-generations 50 ^
  --mutable-region new ^
  --proposal-mode mixed ^
  --entry-rate 0.06 ^
  --table-rate 0.08 ^
  --initial-mode joint_random ^
  --n-random-initial 4096 ^
  --save-winners example_results/lattice_growth_winners_q4_3to4.pkl ^
  --out example_results/lattice_growth_q4_3to4.csv ^
  --plot example_results/fig_lattice_growth_q4_3to4.png
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
from . import blindmicrolattice as BM
from .sharedsquareholonomy import _mutate_rule

FitnessMode = Literal["pretrain", "transport", "two_stage", "flux"]
ProposalMode = Literal["mutate", "mixed", "random"]
MutableRegion = Literal["new", "new_plus_boundary", "all"]


# --------------------------------------------------------------------------- #
# Loading and selecting seed winners
# --------------------------------------------------------------------------- #
def load_micro_lattice_winners(path: str, top_n: int | None = None) -> list[dict]:
    """Load winner records produced by blindmicrolattice.save_winners."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "winners" in obj:
        winners = list(obj["winners"])
    elif isinstance(obj, list):
        winners = list(obj)
    else:
        raise ValueError(f"unrecognized winner file format in {path!r}")
    out = []
    for w in winners:
        if not isinstance(w, dict) or "lattice" not in w:
            continue
        lat = w["lattice"]
        if not hasattr(lat, "nx") or not hasattr(lat, "ny"):
            continue
        out.append(w)
    out.sort(key=lambda r: (
        bool(r.get("metrics", {}).get("full_valid_lattice", False)),
        float(r.get("metrics", {}).get("z2_plaquette_fraction", 0.0)),
        float(r.get("metrics", {}).get("mean_edge_transport", 0.0)),
        float(r.get("fitness", r.get("metrics", {}).get("blind_fitness", 0.0))),
    ), reverse=True)
    if top_n is not None and int(top_n) > 0:
        out = out[:int(top_n)]
    return out


def _parse_size(s: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(s, tuple):
        return int(s[0]), int(s[1])
    t = str(s).strip().lower().replace(" ", "")
    if "x" not in t:
        n = int(t)
        return n, n
    a, b = t.split("x", 1)
    return int(a), int(b)


# --------------------------------------------------------------------------- #
# Embedding a selected lattice as the north-west core of a larger lattice
# --------------------------------------------------------------------------- #
def _copy_rule_if_compatible(old: ML.MicroLattice, new: ML.MicroLattice, old_v: int, new_v: int) -> bool:
    """Copy one local rule table when arity and alphabet match."""
    if int(old.q) != int(new.q):
        return False
    if len(old.joint.preds[int(old_v)]) != len(new.joint.preds[int(new_v)]):
        return False
    old_rule = np.asarray(old.joint.rules[int(old_v)], dtype=np.int64)
    new_rule = np.asarray(new.joint.rules[int(new_v)], dtype=np.int64)
    if old_rule.shape != new_rule.shape:
        return False
    new.joint.rules[int(new_v)] = old_rule.copy()
    return True


def _edge_key(src: tuple[int, int], dst: tuple[int, int]):
    return (tuple(src), tuple(dst))


def transplant_core_rules(old: ML.MicroLattice, new: ML.MicroLattice) -> set[int]:
    """Copy all rule tables belonging to the old lattice into the NW core.

    The microscopic-lattice constructor is coordinate-structured, so semantic
    vertices are copied by coordinate and edge role rather than by raw index.
    Returns the set of new-vertex indices that were inherited from the seed.
    """
    inherited: set[int] = set()
    if int(old.q) != int(new.q) or int(old.w) != int(new.w):
        raise ValueError("old and new lattices must have the same q and w")
    if int(new.nx) < int(old.nx) or int(new.ny) < int(old.ny):
        raise ValueError("new lattice must be at least as large as the old lattice")

    # Corner output vertices in the old rectangle.
    for r in range(int(old.ny) + 1):
        for c in range(int(old.nx) + 1):
            old_vs = old.node_out[(r, c)]
            new_vs = new.node_out[(r, c)]
            for ov, nv in zip(old_vs, new_vs):
                if _copy_rule_if_compatible(old, new, int(ov), int(nv)):
                    inherited.add(int(nv))

    # Edge transporter vertices wholly in the old rectangle.
    for key, old_ed in old.edges.items():
        if key not in new.edges:
            continue
        new_ed = new.edges[key]
        for attr in ("branch_left", "branch_right", "panel_left", "panel_right"):
            for ov, nv in zip(getattr(old_ed, attr), getattr(new_ed, attr)):
                if _copy_rule_if_compatible(old, new, int(ov), int(nv)):
                    inherited.add(int(nv))
        # Preserve flip metadata when present.
        try:
            new_ed.flip = int(old_ed.flip)
        except Exception:
            pass
    return inherited


def grow_lattice(
    seed: ML.MicroLattice,
    to_size: tuple[int, int] = (4, 4),
    rng: np.random.Generator | None = None,
    new_ensemble: str = "random",
) -> tuple[ML.MicroLattice, set[int]]:
    """Embed a seed lattice as the NW core of a larger lattice."""
    if rng is None:
        rng = np.random.default_rng(0)
    nx, ny = int(to_size[0]), int(to_size[1])
    if nx < int(seed.nx) or ny < int(seed.ny):
        raise ValueError("to_size must be no smaller than the seed lattice")
    lat = ML.make_microscopic_lattice(q=int(seed.q), nx=nx, ny=ny, w=int(seed.w), rng=rng, ensemble=new_ensemble)
    inherited = transplant_core_rules(seed, lat)
    lat.meta = dict(lat.meta)
    lat.meta.update(kind="grown_lattice", seed_nx=int(seed.nx), seed_ny=int(seed.ny), grown_nx=nx, grown_ny=ny, new_ensemble=str(new_ensemble))
    return lat, inherited


# --------------------------------------------------------------------------- #
# Region-aware mutation
# --------------------------------------------------------------------------- #
def _new_region_vertices(lat: ML.MicroLattice, old_nx: int, old_ny: int, region: MutableRegion = "new") -> set[int]:
    """Semantic set of vertices allowed to mutate during growth."""
    old_nx = int(old_nx); old_ny = int(old_ny)
    out: set[int] = set()
    if region == "all":
        return {v for v in range(lat.joint.k) if len(lat.joint.preds[v]) > 0}

    # New corner output vertices outside the old NW corner grid.
    for (r, c), vs in lat.node_out.items():
        is_new_node = (int(r) > old_ny or int(c) > old_nx)
        is_boundary_node = (int(r) == old_ny or int(c) == old_nx)
        if is_new_node or (region == "new_plus_boundary" and is_boundary_node):
            out.update(int(v) for v in vs if len(lat.joint.preds[int(v)]) > 0)

    # Edge transporter vertices for any edge not wholly contained in the old lattice.
    for key, ed in lat.edges.items():
        src, dst = key
        wholly_old = (src[0] <= old_ny and src[1] <= old_nx and dst[0] <= old_ny and dst[1] <= old_nx)
        touches_boundary = (src[0] == old_ny or src[1] == old_nx or dst[0] == old_ny or dst[1] == old_nx)
        if (not wholly_old) or (region == "new_plus_boundary" and touches_boundary):
            for attr in ("branch_left", "branch_right", "panel_left", "panel_right"):
                out.update(int(v) for v in getattr(ed, attr) if len(lat.joint.preds[int(v)]) > 0)
    return out


def mutate_growth_lattice(
    lat: ML.MicroLattice,
    rng: np.random.Generator,
    old_nx: int,
    old_ny: int,
    mutable_region: MutableRegion = "new",
    entry_rate: float = 0.06,
    table_rate: float = 0.08,
    force_one: bool = True,
) -> ML.MicroLattice:
    out = BM.clone_lattice(lat)
    q = int(out.q)
    vertices = sorted(_new_region_vertices(out, int(old_nx), int(old_ny), mutable_region))
    touched = 0
    for v in vertices:
        if rng.random() < float(entry_rate):
            out.joint.rules[int(v)] = _mutate_rule(out.joint.rules[int(v)], q, float(table_rate), rng)
            touched += 1
    if force_one and vertices and touched == 0:
        v = int(rng.choice(vertices))
        rate = max(float(table_rate), 1.0 / max(1, int(out.joint.rules[v].size)))
        out.joint.rules[v] = _mutate_rule(out.joint.rules[v], q, rate, rng)
    out.meta = dict(out.meta)
    out.meta["growth_mutated"] = True
    out.meta["mutable_region"] = str(mutable_region)
    return out


# --------------------------------------------------------------------------- #
# Evaluation with core/new-region diagnostics
# --------------------------------------------------------------------------- #
def _region_fractions(meas: dict, old_nx: int, old_ny: int, new_nx: int, new_ny: int) -> dict:
    plaquettes = list(meas.get("plaquettes", []))
    core = []
    growth = []
    for p in plaquettes:
        r, c = int(p.get("r", -1)), int(p.get("c", -1))
        target = core if (r < int(old_ny) and c < int(old_nx)) else growth
        target.append(p)

    def frac(rows, pred):
        return float(sum(1 for x in rows if pred(x)) / max(1, len(rows)))
    z2_pred = lambda p: bool(p.get("valid", False)) and not math.isnan(float(p.get("flux", math.nan)))
    non_pred = lambda p: bool(p.get("valid", False)) and bool(p.get("nontrivial", False))
    return dict(
        core_plaquettes=int(len(core)),
        growth_plaquettes=int(len(growth)),
        core_z2_fraction=frac(core, z2_pred),
        growth_z2_fraction=frac(growth, z2_pred),
        core_nontrivial_fraction=frac([p for p in core if z2_pred(p)], non_pred) if any(z2_pred(p) for p in core) else 0.0,
        growth_nontrivial_fraction=frac([p for p in growth if z2_pred(p)], non_pred) if any(z2_pred(p) for p in growth) else 0.0,
    )


def evaluate_growth_candidate(
    lat: ML.MicroLattice,
    rng: np.random.Generator,
    old_nx: int,
    old_ny: int,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    fitness: FitnessMode = "transport",
    generation: int = 0,
    pretrain_generations: int = 0,
    max_loop_side: int = 4,
) -> dict:
    meas = ML.measure_microscopic_lattice(lat, initial_mode=initial_mode, n_random_initial=int(n_random_initial), rng=rng)
    row: dict = dict(
        q=int(lat.q), w=int(lat.w), nx=int(lat.nx), ny=int(lat.ny), old_nx=int(old_nx), old_ny=int(old_ny),
        valid_plaquette_fraction=float(meas["valid_fraction"]),
        z2_plaquette_fraction=float(meas["z2_fraction"]),
        nontrivial_plaquette_fraction=float(meas["nontrivial_fraction"]),
        delta_type_counts=json.dumps(meas.get("delta_counts", {})),
        group_name_counts=json.dumps(meas.get("group_counts", {})),
    )
    row.update(BM._plaquette_stats(meas, int(lat.q), int(lat.w)))
    row.update(BM._wilson_summary(meas["flux_grid"], max_side=int(max_loop_side)))
    row.update(_region_fractions(meas, int(old_nx), int(old_ny), int(lat.nx), int(lat.ny)))

    if fitness == "pretrain":
        stage = "pretrain"; fit = BM._quality_score(row, int(lat.w))
    elif fitness == "transport":
        stage = "transport"; fit = BM._transport_score(row, int(lat.w))
    elif fitness == "flux":
        stage = "flux"; fit = BM._flux_score(row, int(lat.w))
    elif fitness == "two_stage":
        if int(generation) < int(pretrain_generations):
            stage = "pretrain"; fit = BM._quality_score(row, int(lat.w))
        else:
            stage = "transport"; fit = BM._transport_score(row, int(lat.w))
    else:
        raise ValueError(f"unknown fitness {fitness!r}")
    row.update(blind_fitness=float(fit), blind_fitness_stage=stage)
    row["core_preserved"] = bool(row["core_z2_fraction"] >= 0.999)
    row["growth_complete"] = bool(row["growth_z2_fraction"] >= 0.999)
    row["full_grown_lattice"] = bool(row["z2_plaquette_fraction"] >= 0.999)
    row["selected_growth"] = bool(row["core_preserved"] and row["growth_z2_fraction"] >= 0.80 and row["mean_edge_transport"] >= 0.80)
    row["nontrivial_flux"] = bool(row["nontrivial_plaquette_fraction"] > 0.05)
    return row


# --------------------------------------------------------------------------- #
# Evolutionary curriculum search
# --------------------------------------------------------------------------- #
def _select_parent(scored_pop: list[tuple[float, ML.MicroLattice, dict]], rng: np.random.Generator) -> ML.MicroLattice:
    ordered = sorted(scored_pop, key=lambda t: t[0], reverse=True)
    pool = ordered[:max(1, len(ordered)//2)]
    contestants = rng.choice(len(pool), size=min(3, len(pool)), replace=False)
    return max((pool[int(i)] for i in contestants), key=lambda t: t[0])[1]


def run_lattice_growth(
    seed_winners: str,
    q: int = 4,
    from_size: tuple[int, int] = (3, 3),
    to_size: tuple[int, int] = (4, 4),
    w: int = 1,
    top_n: int = 20,
    runs: int = 10,
    population: int = 40,
    generations: int = 150,
    fitness: FitnessMode = "two_stage",
    pretrain_generations: int = 50,
    mutable_region: MutableRegion = "new",
    proposal_mode: ProposalMode = "mixed",
    entry_rate: float = 0.06,
    table_rate: float = 0.08,
    initial_mode: str = "joint_random",
    n_random_initial: int = 4096,
    max_loop_side: int = 4,
    new_ensemble: str = "random",
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[float, ML.MicroLattice, dict]]]:
    old_nx, old_ny = _parse_size(from_size)
    new_nx, new_ny = _parse_size(to_size)
    loaded = load_micro_lattice_winners(seed_winners, top_n=int(top_n))
    seeds = [r for r in loaded if int(r["lattice"].q) == int(q) and int(r["lattice"].w) == int(w) and int(r["lattice"].nx) == old_nx and int(r["lattice"].ny) == old_ny]
    if not seeds:
        raise ValueError(f"no matching {old_nx}x{old_ny}, q={q}, w={w} seed lattices found in {seed_winners}")

    rows: list[dict] = []
    winners: list[tuple[float, ML.MicroLattice, dict]] = []
    # Baseline evaluation of grown seeds before selection.
    for si, rec in enumerate(seeds):
        rng = np.random.default_rng((int(base_seed)*1000003 + si*1009 + 17) % 2**32)
        grown, inherited = grow_lattice(rec["lattice"], (new_nx, new_ny), rng, new_ensemble=new_ensemble)
        m = evaluate_growth_candidate(grown, rng, old_nx, old_ny, initial_mode, n_random_initial, fitness="transport" if fitness == "two_stage" else fitness, max_loop_side=max_loop_side)
        m.update(search_mode="seed_grown", run=-1, generation=0, candidate=si, seed_index=si, inherited_vertices=len(inherited), mutable_region=str(mutable_region))
        rows.append(m)
        winners.append((float(m["blind_fitness"]), grown, m))
    if verbose:
        print(f"lattice-growth loaded {len(seeds)} seed winners; baseline grown evaluation done", flush=True)

    for run in range(int(runs)):
        seed0 = (int(base_seed)*2000003 + int(q)*20011 + run*9733 + new_nx*101 + new_ny*103) % 2**32
        rng = np.random.default_rng(seed0)
        seed_rec = seeds[int(run) % len(seeds)]
        base_lat, inherited = grow_lattice(seed_rec["lattice"], (new_nx, new_ny), rng, new_ensemble=new_ensemble)
        pop: list[ML.MicroLattice] = [BM.clone_lattice(base_lat)]
        while len(pop) < int(population):
            if len(pop) < max(2, int(0.25 * population)):
                pop.append(mutate_growth_lattice(base_lat, rng, old_nx, old_ny, mutable_region, entry_rate, table_rate))
            else:
                # Occasional independently-grown seeds improve diversity while keeping curriculum origin.
                rec = seeds[int(rng.integers(0, len(seeds)))]
                lat0, _ = grow_lattice(rec["lattice"], (new_nx, new_ny), rng, new_ensemble=new_ensemble)
                pop.append(mutate_growth_lattice(lat0, rng, old_nx, old_ny, mutable_region, entry_rate, table_rate))
        for gen in range(int(generations) + 1):
            scored: list[tuple[float, ML.MicroLattice, dict]] = []
            for ci, cand in enumerate(pop):
                m = evaluate_growth_candidate(cand, rng, old_nx, old_ny, initial_mode, n_random_initial, fitness, gen, pretrain_generations, max_loop_side)
                m.update(search_mode="growth", run=int(run), generation=int(gen), candidate=int(ci), seed=int(seed0), seed_index=int(run) % len(seeds), mutable_region=str(mutable_region), proposal_mode=str(proposal_mode))
                rows.append(m)
                scored.append((float(m["blind_fitness"]), cand, m))
                winners.append((float(m["blind_fitness"]), cand, m))
            best = max(scored, key=lambda t: t[0])
            if verbose:
                bm = best[2]
                print(
                    f"lattice-growth run={run} gen={gen} stage={bm.get('blind_fitness_stage')} "
                    f"best={best[0]:.4f} z2={float(bm.get('z2_plaquette_fraction',0)):.3f} "
                    f"core={float(bm.get('core_z2_fraction',0)):.3f} growth={float(bm.get('growth_z2_fraction',0)):.3f} "
                    f"trans={float(bm.get('mean_edge_transport',0)):.3f} nontriv={float(bm.get('nontrivial_plaquette_fraction',0)):.3f}",
                    flush=True,
                )
            if gen == int(generations):
                break
            ordered = sorted(scored, key=lambda t: t[0], reverse=True)
            elite_n = max(1, int(round(0.15 * int(population))))
            newpop = [BM.clone_lattice(t[1]) for t in ordered[:elite_n]]
            while len(newpop) < int(population):
                if proposal_mode in ("mixed", "random") and rng.random() < (0.05 if proposal_mode == "mixed" else 1.0):
                    rec = seeds[int(rng.integers(0, len(seeds)))]
                    child, _ = grow_lattice(rec["lattice"], (new_nx, new_ny), rng, new_ensemble=new_ensemble)
                    if proposal_mode == "mixed":
                        child = mutate_growth_lattice(child, rng, old_nx, old_ny, mutable_region, entry_rate, table_rate)
                else:
                    parent = _select_parent(scored, rng)
                    child = mutate_growth_lattice(parent, rng, old_nx, old_ny, mutable_region, entry_rate, table_rate)
                newpop.append(child)
            pop = newpop
    return pd.DataFrame(rows), winners


# --------------------------------------------------------------------------- #
# Analysis / plotting / saving
# --------------------------------------------------------------------------- #
def _best_by_run_generation(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df.search_mode == "growth"].copy() if "search_mode" in df else df.iloc[0:0].copy()
    if ev.empty:
        return ev
    idx = ev.groupby(["run", "generation"])["blind_fitness"].idxmax()
    return ev.loc[idx].copy()


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


def analyze_lattice_growth(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    seed = df[df.search_mode == "seed_grown"]
    best = _best_by_run_generation(df)
    final = best.loc[best.groupby("run")["generation"].idxmax()] if len(best) else best
    all_time = df.loc[df["blind_fitness"].idxmax()].to_dict() if len(df) else {}
    final_success = float(final.full_grown_lattice.mean()) if len(final) else math.nan
    final_selected = float(final.selected_growth.mean()) if len(final) else math.nan
    seed_full = float(seed.full_grown_lattice.mean()) if len(seed) else math.nan
    if final_success > 0:
        verdict = "Z2 LATTICE GROWTH SIGNAL: selected 3x3 lattices extend to full valid 4x4 coupled lattices"
    elif final_selected > seed_full:
        verdict = "PARTIAL Z2 LATTICE GROWTH SIGNAL: extension improves but does not reach full validity"
    else:
        verdict = "NO Z2 LATTICE GROWTH SIGNAL in this regime"
    by_gen = []
    if len(best):
        for gen, g in best.groupby("generation"):
            by_gen.append(dict(
                generation=int(gen),
                best_fitness=float(g.blind_fitness.max()),
                mean_best_fitness=float(g.blind_fitness.mean()),
                full_growth_run_fraction=float(g.full_grown_lattice.mean()),
                selected_growth_run_fraction=float(g.selected_growth.mean()),
                mean_z2_fraction=float(g.z2_plaquette_fraction.mean()),
                mean_core_z2_fraction=float(g.core_z2_fraction.mean()),
                mean_growth_z2_fraction=float(g.growth_z2_fraction.mean()),
                best_growth_z2_fraction=float(g.growth_z2_fraction.max()),
                mean_nontrivial_fraction=float(g.nontrivial_plaquette_fraction.mean()),
                best_mean_transport=float(g.mean_edge_transport.max()),
                stage=str(g.blind_fitness_stage.iloc[0]) if "blind_fitness_stage" in g else "",
            ))
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        seed_grown_count=int(len(seed)),
        seed_full_grown_fraction=seed_full,
        seed_mean_core_z2=float(seed.core_z2_fraction.mean()) if len(seed) else math.nan,
        seed_mean_growth_z2=float(seed.growth_z2_fraction.mean()) if len(seed) else math.nan,
        final_run_count=int(len(final)),
        final_full_grown_fraction=final_success,
        final_selected_growth_fraction=final_selected,
        final_mean_core_z2=float(final.core_z2_fraction.mean()) if len(final) else math.nan,
        final_mean_growth_z2=float(final.growth_z2_fraction.mean()) if len(final) else math.nan,
        final_mean_nontrivial_flux=float(final[final.full_grown_lattice.astype(bool)].nontrivial_plaquette_fraction.mean()) if len(final[final.full_grown_lattice.astype(bool)]) else 0.0,
        all_time_best_fitness=float(all_time.get("blind_fitness", math.nan)),
        all_time_full_grown=bool(all_time.get("full_grown_lattice", False)),
        all_time_z2_fraction=float(all_time.get("z2_plaquette_fraction", math.nan)),
        all_time_core_z2=float(all_time.get("core_z2_fraction", math.nan)),
        all_time_growth_z2=float(all_time.get("growth_z2_fraction", math.nan)),
        all_time_nontrivial_fraction=float(all_time.get("nontrivial_plaquette_fraction", math.nan)),
        all_time_mean_transport=float(all_time.get("mean_edge_transport", math.nan)),
        final_group_name_counts=_merged_json_counts(final.get("group_name_counts", pd.Series(dtype=str))) if len(final) else {},
        final_delta_type_counts=_merged_json_counts(final.get("delta_type_counts", pd.Series(dtype=str))) if len(final) else {},
        best_by_generation=by_gen,
    )


def save_winners(path: str, winners: list[tuple[float, ML.MicroLattice, dict]], top_n: int = 50, min_full: bool = False) -> None:
    ranked = sorted(winners, key=lambda t: (
        bool(t[2].get("full_grown_lattice", False)),
        bool(t[2].get("selected_growth", False)),
        float(t[2].get("z2_plaquette_fraction", 0.0)),
        float(t[2].get("growth_z2_fraction", 0.0)),
        float(t[2].get("blind_fitness", t[0])),
        float(t[2].get("nontrivial_plaquette_fraction", 0.0)),
    ), reverse=True)
    out = []
    for fit, lat, row in ranked:
        if min_full and not bool(row.get("full_grown_lattice", False)):
            continue
        out.append(dict(fitness=float(fit), metrics=dict(row), lattice=lat))
        if len(out) >= int(top_n):
            break
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(kind="lattice_growth_winners", winners=out), f)


def plot_lattice_growth(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); return
    best = _best_by_run_generation(df)
    if len(best):
        g = best.groupby("generation")
        ax[0].plot(g.blind_fitness.max().index, g.blind_fitness.max().values, marker="o", label="global best")
        ax[0].plot(g.blind_fitness.mean().index, g.blind_fitness.mean().values, marker="o", label="mean run-best")
        ax[0].set_title("4x4 growth fitness")
        ax[0].set_xlabel("generation"); ax[0].set_ylabel("fitness"); ax[0].legend()
        ax[1].plot(g.full_grown_lattice.mean().index, g.full_grown_lattice.mean().values, marker="o", label="full 4x4")
        ax[1].plot(g.selected_growth.mean().index, g.selected_growth.mean().values, marker="o", label="selected growth")
        ax[1].plot(g.growth_z2_fraction.mean().index, g.growth_z2_fraction.mean().values, marker="o", label="mean new plaquettes")
        ax[1].set_title("curriculum extension over time")
        ax[1].set_xlabel("generation"); ax[1].set_ylabel("fraction"); ax[1].legend()
        sc = ax[2].scatter(best.core_z2_fraction, best.growth_z2_fraction, c=best.nontrivial_plaquette_fraction, s=20)
        ax[2].set_title("core preserved vs new growth")
        ax[2].set_xlabel("core Z2 fraction"); ax[2].set_ylabel("new plaquette Z2 fraction")
        fig.colorbar(sc, ax=ax[2], label="nontrivial flux fraction")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Curriculum growth from selected 3x3 Z2 micro-lattice winners to larger coupled lattices.")
    ap.add_argument("q", type=int, nargs="?", default=4)
    ap.add_argument("--seed-winners", required=True)
    ap.add_argument("--from-size", default="3x3")
    ap.add_argument("--to-size", default="4x4")
    ap.add_argument("--w", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--fitness", choices=["pretrain", "transport", "two_stage", "flux"], default="two_stage")
    ap.add_argument("--pretrain-generations", type=int, default=50)
    ap.add_argument("--mutable-region", choices=["new", "new_plus_boundary", "all"], default="new")
    ap.add_argument("--proposal-mode", choices=["mutate", "mixed", "random"], default="mixed")
    ap.add_argument("--entry-rate", type=float, default=0.06)
    ap.add_argument("--table-rate", type=float, default=0.08)
    ap.add_argument("--initial-mode", choices=["source_all", "joint_random"], default="joint_random")
    ap.add_argument("--n-random-initial", type=int, default=4096)
    ap.add_argument("--max-loop-side", type=int, default=4)
    ap.add_argument("--new-ensemble", default="random")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--save-winners", default="")
    ap.add_argument("--save-winner-top-n", type=int, default=50)
    ap.add_argument("--save-winner-min-full", action="store_true")
    ap.add_argument("--out", default="example_results/lattice_growth_q4_3to4.csv")
    ap.add_argument("--plot", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df, winners = run_lattice_growth(
        seed_winners=args.seed_winners,
        q=int(args.q),
        from_size=_parse_size(args.from_size),
        to_size=_parse_size(args.to_size),
        w=int(args.w),
        top_n=int(args.top_n),
        runs=int(args.runs),
        population=int(args.population),
        generations=int(args.generations),
        fitness=args.fitness,
        pretrain_generations=int(args.pretrain_generations),
        mutable_region=args.mutable_region,
        proposal_mode=args.proposal_mode,
        entry_rate=float(args.entry_rate),
        table_rate=float(args.table_rate),
        initial_mode=args.initial_mode,
        n_random_initial=int(args.n_random_initial),
        max_loop_side=int(args.max_loop_side),
        new_ensemble=str(args.new_ensemble),
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = analyze_lattice_growth(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print(json.dumps(summary, indent=2, allow_nan=True))
    if args.plot:
        plot_lattice_growth(df, args.plot)
        print(f"wrote {args.plot}")
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    if args.save_winners:
        save_winners(args.save_winners, winners, top_n=int(args.save_winner_top_n), min_full=bool(args.save_winner_min_full))
        print(f"wrote {args.save_winners}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
