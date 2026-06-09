"""
blindtemporalchain.py -- blind search for transported live labels in a temporal chain.

Purpose
-------
``temporalchain.py`` showed that hand-structured rule sectors (copy,
permutive, block, canalizing) can transport a live selected label through two
successive diamond convergences, while random rules do not.  This module asks
the blind-selection version of that question:

    If the search objective only knows about temporal transport of exact
    interface labels, can it discover rule tables that carry a live label
    through the chain?

The topology is the two-diamond chain implemented in ``temporalchain.py``:

        B0
       /  \
     C1    D1
       \  /
        A1
       /  \
     C2    D2
       \  /
        A2

For each candidate we compute exact equalizer labels S1 at A1 and S2 at A2,
then score transport and source-liveness:

    min(log_q|S1|, log_q|S2|),
    I(S1;S2)/H(S2),
    I(source;S2)/H(S2),
    best deterministic S1 -> S2 accuracy.

No copy/permutation/block target is ever used as fitness.  Copy accuracy and
interface mutual information are not part of the objective; the temporal-chain
transport scores are the objective.

Recommended first run
---------------------
python -m relgauge.blindtemporalchain 4 ^
  --ks-B 2 --ks-M 2 --ks-A 2 --ws 1 ^
  --null-samples 200 --runs 5 --population 20 --generations 50 ^
  --fitness two_stage --pretrain-generations 20 ^
  --target interface --proposal-mode mixed ^
  --entry-rate 0.08 --table-rate 0.10 ^
  --initial-mode joint_random --n-random-initial 4096 ^
  --out example_results/blind_temporal_chain_q4.csv ^
  --plot example_results/fig_blind_temporal_chain_q4.png
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

from . import core as C
from . import temporalchain as T

TargetMode = Literal["all", "interface", "source", "branch", "sink1", "sink2", "carriers"]
ProposalMode = Literal["entry", "table", "mixed"]
FitnessMode = Literal["transport", "two_stage"]
FitnessStage = Literal["pretrain", "transport"]


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


# --------------------------------------------------------------------------- #
# Candidate copying / mutation
# --------------------------------------------------------------------------- #
def clone_temporal_chain(tc: T.TemporalChain) -> T.TemporalChain:
    """Deep-copy a ``TemporalChain`` candidate, including rule tables."""
    joint = C.RelationalSystem(
        tc.joint.k,
        tc.joint.q,
        [tuple(p) for p in tc.joint.preds],
        [np.asarray(r, dtype=np.int64).copy() for r in tc.joint.rules],
        meta=dict(tc.joint.meta),
    )
    return T.TemporalChain(
        joint=joint,
        kB=tc.kB,
        kM=tc.kM,
        kA=tc.kA,
        q=tc.q,
        w=tc.w,
        offB=tc.offB,
        offC1=tc.offC1,
        offD1=tc.offD1,
        offA1=tc.offA1,
        offC2=tc.offC2,
        offD2=tc.offD2,
        offA2=tc.offA2,
        interface_BC1=tuple(tc.interface_BC1),
        interface_BD1=tuple(tc.interface_BD1),
        interface_C1A1=tuple(tc.interface_C1A1),
        interface_D1A1=tuple(tc.interface_D1A1),
        interface_A1C2=tuple(tc.interface_A1C2),
        interface_A1D2=tuple(tc.interface_A1D2),
        interface_C2A2=tuple(tc.interface_C2A2),
        interface_D2A2=tuple(tc.interface_D2A2),
        meta=dict(tc.meta),
    )


def target_vertices(tc: T.TemporalChain, target: TargetMode = "interface") -> tuple[int, ...]:
    """Vertices whose rule tables may mutate in blind search.

    ``interface`` is the practical default.  It mutates the declared transport
    pipeline and the B boundary source, but it does not impose any copy-like
    structure.  ``all`` is the stricter whole-rule search.
    """
    if target == "all":
        return tuple(range(tc.k_total))
    source = [tc.offB + j for j in range(tc.w)]
    branch = []
    branch.extend(t for _s, t in tc.interface_BC1)
    branch.extend(t for _s, t in tc.interface_BD1)
    branch.extend(t for _s, t in tc.interface_A1C2)
    branch.extend(t for _s, t in tc.interface_A1D2)
    sink1 = []
    sink1.extend(t for _s, t in tc.interface_C1A1)
    sink1.extend(t for _s, t in tc.interface_D1A1)
    sink2 = []
    sink2.extend(t for _s, t in tc.interface_C2A2)
    sink2.extend(t for _s, t in tc.interface_D2A2)
    if target == "source":
        return tuple(sorted(set(int(x) for x in source)))
    if target == "branch":
        return tuple(sorted(set(int(x) for x in branch)))
    if target == "sink1":
        return tuple(sorted(set(int(x) for x in sink1)))
    if target == "sink2":
        return tuple(sorted(set(int(x) for x in sink2)))
    if target == "carriers":
        out = []
        out.extend(t for _s, t in tc.interface_A1C2)
        out.extend(t for _s, t in tc.interface_A1D2)
        out.extend(t for _s, t in tc.interface_C2A2)
        out.extend(t for _s, t in tc.interface_D2A2)
        return tuple(sorted(set(int(x) for x in out)))
    if target == "interface":
        out = []
        out.extend(source)
        out.extend(branch)
        out.extend(sink1)
        out.extend(sink2)
        return tuple(sorted(set(int(x) for x in out)))
    raise ValueError(f"unknown target mode: {target!r}")


def _random_rule(q: int, indeg: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, int(q), size=int(q) ** int(indeg), dtype=np.int64)


def _mutate_entries(rule: np.ndarray, q: int, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    out = np.asarray(rule, dtype=np.int64).copy()
    if out.size == 0 or rate <= 0:
        return out, 0
    mask = rng.random(out.size) < float(rate)
    n = int(mask.sum())
    if n <= 0:
        return out, 0
    jumps = rng.integers(1, int(q), size=n, dtype=np.int64)
    out[mask] = (out[mask] + jumps) % int(q)
    return out, n


def mutate_temporal_chain(
    tc: T.TemporalChain,
    rng: np.random.Generator,
    target: TargetMode = "interface",
    proposal_mode: ProposalMode = "mixed",
    entry_rate: float = 0.08,
    table_rate: float = 0.10,
    n_table_mutations: int = 1,
    ensure_change: bool = True,
) -> tuple[T.TemporalChain, dict]:
    child = clone_temporal_chain(tc)
    verts = list(target_vertices(child, target))
    if not verts:
        raise ValueError("empty mutation target")
    proposal_mode = str(proposal_mode)  # type: ignore[assignment]
    if proposal_mode == "mixed":
        proposal_mode = "table" if rng.random() < float(table_rate) else "entry"
    n_entry = 0
    n_table = 0
    changed = False
    if proposal_mode == "table":
        nmut = max(1, int(n_table_mutations))
        for _ in range(nmut):
            v = int(rng.choice(verts))
            child.joint.rules[v] = _random_rule(child.q, len(child.joint.preds[v]), rng)
            n_table += 1
            changed = True
    elif proposal_mode == "entry":
        for v in verts:
            new, n = _mutate_entries(child.joint.rules[int(v)], child.q, entry_rate, rng)
            if n:
                child.joint.rules[int(v)] = new
                n_entry += n
                changed = True
        if ensure_change and not changed:
            v = int(rng.choice(verts))
            rule = child.joint.rules[v].copy()
            idx = int(rng.integers(0, len(rule)))
            rule[idx] = (rule[idx] + int(rng.integers(1, child.q))) % child.q
            child.joint.rules[v] = rule
            n_entry += 1
            changed = True
    else:
        raise ValueError("proposal_mode must be entry, table, or mixed")
    child.joint.meta["rule_ensemble"] = "blind"
    child.joint.meta["blind_target"] = str(target)
    child.meta.update(rule_ensemble="blind", blind_target=str(target))
    return child, dict(target=str(target), proposal_mode=str(proposal_mode), entry_changes=int(n_entry), table_changes=int(n_table))


# --------------------------------------------------------------------------- #
# Evaluation / fitness
# --------------------------------------------------------------------------- #
def evaluate_candidate(
    tc: T.TemporalChain,
    initial_mode: T.InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    rng: np.random.Generator | None = None,
    stage: FitnessStage = "transport",
    transport_threshold: float = 0.80,
    source_threshold: float = 0.80,
    accuracy_threshold: float = 0.95,
) -> dict:
    m = T.temporal_chain_measure(
        tc,
        initial_mode=initial_mode,
        n_random_initial=int(n_random_initial),
        source_for_liveness=source_for_liveness,
        rng=rng,
    )
    min_residual = float(min(m["s1_residual_qcoords"], m["s2_residual_qcoords"]))
    transport = float(m["transport_mi_over_Hs2"])
    source = float(m["source_to_s2_mi_over_Hs2"])
    acc = float(m["s1_to_s2_best_accuracy"])
    exact_ok = bool(m["s1_residual_qcoords"] > 1e-9 and m["s2_residual_qcoords"] > 1e-9)
    live_transport = bool(exact_ok and transport >= float(transport_threshold) and source >= float(source_threshold) and acc >= float(accuracy_threshold))
    # A smooth pretraining objective can climb toward labels before exact live
    # transport exists.  The final stage makes exact/live transport dominant.
    pretrain_score = float(0.35 * min_residual + 0.35 * transport + 0.30 * source + 1e-3 * acc)
    transport_score = float(min_residual + 0.50 * transport + 0.50 * source + 1e-6 * acc)
    score = pretrain_score if stage == "pretrain" else transport_score
    out = dict(m)
    out.update(
        min_residual_qcoords=min_residual,
        blind_transport_score=transport_score,
        blind_pretrain_score=pretrain_score,
        blind_fitness=float(score),
        blind_fitness_stage=str(stage),
        selected_blind_live_transport=bool(live_transport),
        exact_transport_gate=bool(exact_ok),
        transport_threshold=float(transport_threshold),
        source_threshold=float(source_threshold),
        accuracy_threshold=float(accuracy_threshold),
    )
    return out


def _stable_seed(*items: object) -> int:
    s = 0
    for item in items:
        txt = str(item)
        for i, ch in enumerate(txt):
            s = (s * 1315423911 + (i + 1) * ord(ch) + 0x9E3779B9) & 0xFFFFFFFF
    return int(s)


def make_random_candidate(kB: int, kM: int, kA: int, q: int, w: int, rng: np.random.Generator) -> T.TemporalChain:
    tc = T.make_temporal_chain(kB, kM, kA, q, w, rng)
    tc.joint.meta["rule_ensemble"] = "blind_random"
    tc.meta.update(rule_ensemble="blind_random")
    return tc


# --------------------------------------------------------------------------- #
# Search routines
# --------------------------------------------------------------------------- #
def run_null_sample(
    kB: int,
    kM: int,
    kA: int,
    w: int,
    q: int,
    n_samples: int = 100,
    initial_mode: T.InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    transport_threshold: float = 0.80,
    source_threshold: float = 0.80,
    accuracy_threshold: float = 0.95,
    base_seed: int = 0,
) -> tuple[list[dict], list[T.TemporalChain]]:
    rows: list[dict] = []
    candidates: list[T.TemporalChain] = []
    for i in range(int(n_samples)):
        seed = (_stable_seed("null", base_seed, kB, kM, kA, w, q, i)) % 2**32
        rng = np.random.default_rng(seed)
        tc = make_random_candidate(kB, kM, kA, q, w, rng)
        m = evaluate_candidate(
            tc,
            initial_mode=initial_mode,
            n_random_initial=n_random_initial,
            source_for_liveness=source_for_liveness,
            rng=rng,
            stage="transport",
            transport_threshold=transport_threshold,
            source_threshold=source_threshold,
            accuracy_threshold=accuracy_threshold,
        )
        m.update(search_mode="null", run=-1, generation=0, candidate=i, seed=int(seed))
        rows.append(m)
        candidates.append(tc)
    return rows, candidates


def run_evolution(
    kB: int,
    kM: int,
    kA: int,
    w: int,
    q: int,
    runs: int = 5,
    population: int = 16,
    generations: int = 50,
    fitness_mode: FitnessMode = "two_stage",
    pretrain_generations: int = 20,
    target: TargetMode = "interface",
    proposal_mode: ProposalMode = "mixed",
    entry_rate: float = 0.08,
    table_rate: float = 0.10,
    n_table_mutations: int = 1,
    elite_fraction: float = 0.25,
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
        rng = np.random.default_rng(_stable_seed("evolve", base_seed, kB, kM, kA, w, q, run) % 2**32)
        pop: list[T.TemporalChain] = [make_random_candidate(kB, kM, kA, q, w, rng) for _ in range(int(population))]
        for gen in range(int(generations) + 1):
            if fitness_mode == "two_stage":
                stage: FitnessStage = "pretrain" if gen < int(pretrain_generations) else "transport"
            elif fitness_mode == "transport":
                stage = "transport"
            else:
                raise ValueError("fitness_mode must be transport or two_stage")
            evals: list[tuple[float, dict, T.TemporalChain]] = []
            for ci, cand in enumerate(pop):
                # Use a generation-specific RNG for initial-state subsampling so
                # selection cannot overfit a single subset too strongly.
                erng = np.random.default_rng(_stable_seed("eval", base_seed, run, gen, ci) % 2**32)
                m = evaluate_candidate(
                    cand,
                    initial_mode=initial_mode,
                    n_random_initial=n_random_initial,
                    source_for_liveness=source_for_liveness,
                    rng=erng,
                    stage=stage,
                    transport_threshold=transport_threshold,
                    source_threshold=source_threshold,
                    accuracy_threshold=accuracy_threshold,
                )
                m.update(
                    search_mode="evolve",
                    run=int(run),
                    generation=int(gen),
                    candidate=int(ci),
                    target_mode=str(target),
                    proposal_mode=str(proposal_mode),
                    fitness_mode=str(fitness_mode),
                )
                rows.append(m)
                evals.append((float(m["blind_fitness"]), m, cand))
            evals.sort(key=lambda t: t[0], reverse=True)
            best_score, best_m, _best_c = evals[0]
            if verbose:
                print(
                    "blind-temporal run={run} gen={gen} stage={stage} "
                    "best={score:.4f} minres={minres:.4f} trans={trans:.4f} src={src:.4f} live={live}".format(
                        run=run,
                        gen=gen,
                        stage=stage,
                        score=best_score,
                        minres=best_m["min_residual_qcoords"],
                        trans=best_m["transport_mi_over_Hs2"],
                        src=best_m["source_to_s2_mi_over_Hs2"],
                        live=bool(best_m["selected_blind_live_transport"]),
                    ),
                    flush=True,
                )
            # Keep live winners and also top exact/transport candidates for audit.
            for score, m, cand in evals[: max(1, min(4, len(evals)))]:
                if bool(m.get("selected_blind_live_transport", False)) or float(m.get("min_residual_qcoords", 0.0)) > 0:
                    saved.append((float(score), dict(m), clone_temporal_chain(cand)))
            if gen >= int(generations):
                break
            n_elite = max(1, int(math.ceil(float(elite_fraction) * len(pop))))
            elites = [clone_temporal_chain(c) for _s, _m, c in evals[:n_elite]]
            new_pop: list[T.TemporalChain] = [clone_temporal_chain(c) for c in elites]
            while len(new_pop) < int(population):
                parent = elites[int(rng.integers(0, len(elites)))]
                child, _diag = mutate_temporal_chain(
                    parent,
                    rng,
                    target=target,
                    proposal_mode=proposal_mode,
                    entry_rate=entry_rate,
                    table_rate=table_rate,
                    n_table_mutations=n_table_mutations,
                    ensure_change=True,
                )
                new_pop.append(child)
            pop = new_pop
    return rows, saved


def run_blind_temporal_chain(
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2,),
    ws: Iterable[int] = (1,),
    q: int = 4,
    null_samples: int = 100,
    runs: int = 5,
    population: int = 16,
    generations: int = 50,
    fitness_mode: FitnessMode = "two_stage",
    pretrain_generations: int = 20,
    target: TargetMode = "interface",
    proposal_mode: ProposalMode = "mixed",
    entry_rate: float = 0.08,
    table_rate: float = 0.10,
    n_table_mutations: int = 1,
    initial_mode: T.InitialMode = "joint_random",
    n_random_initial: int = 4096,
    source_for_liveness: Literal["initial", "postB"] = "postB",
    transport_threshold: float = 0.80,
    source_threshold: float = 0.80,
    accuracy_threshold: float = 0.95,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[float, dict, T.TemporalChain]]]:
    rows: list[dict] = []
    winners: list[tuple[float, dict, T.TemporalChain]] = []
    for kB in ks_B:
        for kM in ks_M:
            for kA in ks_A:
                for w in ws:
                    if w > min(kB, kM) or kA < 2 * w:
                        continue
                    nr, ncands = run_null_sample(
                        kB=int(kB), kM=int(kM), kA=int(kA), w=int(w), q=int(q),
                        n_samples=int(null_samples),
                        initial_mode=initial_mode,
                        n_random_initial=int(n_random_initial),
                        source_for_liveness=source_for_liveness,
                        transport_threshold=float(transport_threshold),
                        source_threshold=float(source_threshold),
                        accuracy_threshold=float(accuracy_threshold),
                        base_seed=int(base_seed),
                    )
                    rows.extend(nr)
                    # retain unusually good null candidates for calibration audit
                    for r, c in zip(nr, ncands):
                        if bool(r.get("selected_blind_live_transport", False)):
                            winners.append((float(r["blind_fitness"]), dict(r), clone_temporal_chain(c)))
                    if verbose:
                        print(f"blind-temporal null kB={kB}, kM={kM}, kA={kA}, w={w} done", flush=True)
                    erows, ewinners = run_evolution(
                        kB=int(kB), kM=int(kM), kA=int(kA), w=int(w), q=int(q),
                        runs=int(runs),
                        population=int(population),
                        generations=int(generations),
                        fitness_mode=fitness_mode,
                        pretrain_generations=int(pretrain_generations),
                        target=target,
                        proposal_mode=proposal_mode,
                        entry_rate=float(entry_rate),
                        table_rate=float(table_rate),
                        n_table_mutations=int(n_table_mutations),
                        initial_mode=initial_mode,
                        n_random_initial=int(n_random_initial),
                        source_for_liveness=source_for_liveness,
                        transport_threshold=float(transport_threshold),
                        source_threshold=float(source_threshold),
                        accuracy_threshold=float(accuracy_threshold),
                        base_seed=int(base_seed),
                        verbose=verbose,
                    )
                    rows.extend(erows)
                    winners.extend(ewinners)
                    if verbose:
                        print(f"blind-temporal evolve kB={kB}, kM={kM}, kA={kA}, w={w} done", flush=True)
    df = pd.DataFrame(rows)
    return df, winners


# --------------------------------------------------------------------------- #
# Analysis / plotting / persistence
# --------------------------------------------------------------------------- #
def analyze_blind_temporal_chain(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    null = df[df["search_mode"] == "null"]
    evol = df[df["search_mode"] == "evolve"]
    final = evol[evol["generation"] == evol["generation"].max()] if len(evol) else evol
    null_live = float(null["selected_blind_live_transport"].mean()) if len(null) else math.nan
    null_best = float(null["blind_fitness"].max()) if len(null) else math.nan
    null_best_minres = float(null["min_residual_qcoords"].max()) if len(null) else math.nan
    null_best_transport = float(null["transport_mi_over_Hs2"].max()) if len(null) else math.nan
    null_best_source = float(null["source_to_s2_mi_over_Hs2"].max()) if len(null) else math.nan
    final_live = float(final.groupby("run")["selected_blind_live_transport"].max().mean()) if len(final) else math.nan
    final_best = float(final.groupby("run")["blind_fitness"].max().mean()) if len(final) else math.nan
    final_best_minres = float(final.groupby("run")["min_residual_qcoords"].max().mean()) if len(final) else math.nan
    final_best_transport = float(final.groupby("run")["transport_mi_over_Hs2"].max().mean()) if len(final) else math.nan
    final_best_source = float(final.groupby("run")["source_to_s2_mi_over_Hs2"].max().mean()) if len(final) else math.nan
    all_best = evol.sort_values("blind_fitness", ascending=False).head(1).iloc[0].to_dict() if len(evol) else {}
    if not math.isnan(final_live) and final_live > 0.5 and (math.isnan(null_live) or null_live < 0.1):
        verdict = "BLIND TEMPORAL LIVE-TRANSPORT SIGNAL: search discovers transported live labels above the random null"
    elif not math.isnan(final_best_transport) and final_best_transport > 0.6 and final_best_source > 0.6:
        verdict = "PARTIAL BLIND TEMPORAL TRANSPORT SIGNAL: transport/source information improves but selection is not robust"
    elif not math.isnan(final_best_minres) and final_best_minres > (0 if math.isnan(null_best_minres) else null_best_minres) + 0.2:
        verdict = "EXACT TEMPORAL LABEL SIGNAL WITHOUT STRONG LIVE TRANSPORT"
    else:
        verdict = "NO BLIND TEMPORAL LIVE-TRANSPORT SIGNAL in this regime"
    best_by_generation = []
    if len(evol):
        for gen, sub in evol.groupby("generation"):
            best_by_generation.append(
                dict(
                    generation=int(gen),
                    best_fitness=float(sub["blind_fitness"].max()),
                    mean_best_fitness=float(sub.groupby("run")["blind_fitness"].max().mean()),
                    live_run_fraction=float(sub.groupby("run")["selected_blind_live_transport"].max().mean()),
                    best_min_residual=float(sub["min_residual_qcoords"].max()),
                    best_transport=float(sub["transport_mi_over_Hs2"].max()),
                    best_source_to_s2=float(sub["source_to_s2_mi_over_Hs2"].max()),
                    stage=str(sub["blind_fitness_stage"].iloc[0]),
                )
            )
    top_cols = [
        "search_mode", "run", "generation", "candidate", "blind_fitness", "blind_fitness_stage",
        "min_residual_qcoords", "s1_residual_qcoords", "s2_residual_qcoords",
        "transport_mi_over_Hs2", "source_to_s2_mi_over_Hs2", "s1_to_s2_best_accuracy",
        "selected_blind_live_transport", "target_mode", "proposal_mode", "fitness_mode",
    ]
    top_candidates = []
    if len(evol):
        top = evol.sort_values(["selected_blind_live_transport", "blind_fitness"], ascending=[False, False]).head(20)
        for _, row in top.iterrows():
            top_candidates.append({c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in top_cols if c in row.index})
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        null_live_transport_fraction=null_live,
        null_best_fitness=null_best,
        null_best_min_residual_qcoords=null_best_minres,
        null_best_transport_mi_over_Hs2=null_best_transport,
        null_best_source_to_s2_mi_over_Hs2=null_best_source,
        evolved_final_live_run_fraction=final_live,
        evolved_final_best_fitness_mean=final_best,
        evolved_final_best_min_residual_qcoords=final_best_minres,
        evolved_final_best_transport_mi_over_Hs2=final_best_transport,
        evolved_final_best_source_to_s2_mi_over_Hs2=final_best_source,
        all_time_best={k: (v.item() if hasattr(v, "item") else v) for k, v in all_best.items()} if all_best else {},
        best_by_generation=best_by_generation,
        top_candidates=top_candidates,
    )


def plot_blind_temporal_chain(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evol = df[df["search_mode"] == "evolve"]
    null = df[df["search_mode"] == "null"]
    if evol.empty:
        raise ValueError("no evolve rows to plot")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    g = evol.groupby("generation")
    ax[0].plot(g["blind_fitness"].max().index, g["blind_fitness"].max().values, "o-", label="global best")
    ax[0].plot(g.apply(lambda s: s.groupby("run")["blind_fitness"].max().mean()).index,
               g.apply(lambda s: s.groupby("run")["blind_fitness"].max().mean()).values,
               "o-", label="mean run-best")
    if len(null):
        ax[0].axhline(float(null["blind_fitness"].max()), linestyle=":", label="null best")
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("blind temporal fitness")
    ax[0].set_title("transport fitness selected by consistency")
    ax[0].legend(fontsize=8)

    live = g.apply(lambda s: s.groupby("run")["selected_blind_live_transport"].max().mean())
    ax[1].plot(live.index, live.values, "o-")
    ax[1].set_xlabel("generation")
    ax[1].set_ylabel("fraction among run-bests")
    ax[1].set_title("blind live-transport over time")

    ax[2].scatter(evol["source_to_s2_mi_over_Hs2"], evol["transport_mi_over_Hs2"], s=14, alpha=0.45)
    ax[2].set_xlabel(r"$I(source;S_2)/H(S_2)$")
    ax[2].set_ylabel(r"$I(S_1;S_2)/H(S_2)$")
    ax[2].set_title("source-liveness vs transport")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_winners(path: str, winners: list[tuple[float, dict, T.TemporalChain]], top_n: int = 30, min_score: float | None = None) -> None:
    ranked = sorted(winners, key=lambda t: t[0], reverse=True)
    if min_score is not None:
        ranked = [x for x in ranked if x[0] >= float(min_score)]
    ranked = ranked[: int(top_n)]
    payload = {
        "kind": "blind_temporal_chain_winners",
        "version": 1,
        "n_winners": len(ranked),
        "winners": [dict(score=float(score), metrics=dict(metrics), candidate=cand) for score, metrics, cand in ranked],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("q", type=int, nargs="?", default=4)
    ap.add_argument("--ks-B", default="2")
    ap.add_argument("--ks-M", default="2")
    ap.add_argument("--ks-A", default="2")
    ap.add_argument("--ws", default="1")
    ap.add_argument("--null-samples", type=int, default=100)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--fitness", choices=("transport", "two_stage"), default="two_stage")
    ap.add_argument("--pretrain-generations", type=int, default=20)
    ap.add_argument("--target", choices=("all", "interface", "source", "branch", "sink1", "sink2", "carriers"), default="interface")
    ap.add_argument("--proposal-mode", choices=("entry", "table", "mixed"), default="mixed")
    ap.add_argument("--entry-rate", type=float, default=0.08)
    ap.add_argument("--table-rate", type=float, default=0.10)
    ap.add_argument("--n-table-mutations", type=int, default=1)
    ap.add_argument("--initial-mode", choices=("source_all", "source_random", "joint_random"), default="joint_random")
    ap.add_argument("--n-random-initial", type=int, default=4096)
    ap.add_argument("--source-for-liveness", choices=("initial", "postB"), default="postB")
    ap.add_argument("--transport-threshold", type=float, default=0.80)
    ap.add_argument("--source-threshold", type=float, default=0.80)
    ap.add_argument("--accuracy-threshold", type=float, default=0.95)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--save-winners", default=None)
    ap.add_argument("--save-winner-top-n", type=int, default=30)
    ap.add_argument("--save-winner-min-score", type=float, default=None)
    ap.add_argument("--out", default="example_results/blind_temporal_chain.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df, winners = run_blind_temporal_chain(
        ks_B=_parse_ints(args.ks_B),
        ks_M=_parse_ints(args.ks_M),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        q=int(args.q),
        null_samples=int(args.null_samples),
        runs=int(args.runs),
        population=int(args.population),
        generations=int(args.generations),
        fitness_mode=args.fitness,  # type: ignore[arg-type]
        pretrain_generations=int(args.pretrain_generations),
        target=args.target,  # type: ignore[arg-type]
        proposal_mode=args.proposal_mode,  # type: ignore[arg-type]
        entry_rate=float(args.entry_rate),
        table_rate=float(args.table_rate),
        n_table_mutations=int(args.n_table_mutations),
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
    summary = analyze_blind_temporal_chain(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    if args.save_winners:
        save_winners(args.save_winners, winners, top_n=int(args.save_winner_top_n), min_score=args.save_winner_min_score)
        summary["saved_winners"] = args.save_winners
        summary["saved_winners_count_requested"] = int(args.save_winner_top_n)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"wrote {args.save_winners}")
    if args.plot:
        plot_blind_temporal_chain(df, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2, default=float))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
