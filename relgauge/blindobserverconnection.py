"""
blindobserverconnection.py

Blind observer-relative connection selection.

This module is the selection companion to observerframebundle.py.  It keeps the
closed-system/observer-relative philosophy intact:

    - observers are proper SCC subsystems;
    - their frames are extracted from live boundary-port quotients;
    - inter-observer transports are induced by exact channel common factors;
    - diamonds/cycles are discovered from the observer cut graph;
    - the selection objective rewards only liveness, deterministic transport,
      branch completion and loop-automorphism validity of those induced objects.

The v4 default uses observer-frame ``local_charts`` rather than shared edge
common-factor coordinates.  Endpoint labels are canonicalized independently on
the two sides of each boundary channel, so an oriented flip edge remains a flip
in the connection instead of being renamed away by a shared equalizer label.
The score still does not reward path equality/flatness.  It rewards that two
completed paths close as a well-defined automorphism of the source frame;
whether that automorphism is identity or nontrivial is audited post-hoc.

It does NOT reward a target group, C2, nontrivial holonomy, a flux value, a
particle label, or a chosen matter sector.  Group and holonomy are audited only
post-hoc by observerframebundle.py.

The default controlled topology is a four-observer diamond with random local
rules (frame_random_diamond).  This mirrors the older shared-square logic while
moving the comparison from external corners/edges to observer-relative SCC
frames.  More exploratory graph modes (componented, er) are also supported, but
random-start discovery there is expected to be much harder because usable
observer diamonds are not guaranteed.

CLI example
-----------
python -m relgauge.blindobserverconnection 4 \
  --graph-mode frame_random_diamond --population 24 --generations 40 \
  --mutation-rate 0.08 --out example_results/blind_observer_connection.csv \
  --winners example_results/blind_observer_connection_winners.pkl

Seeded recovery / positive control:

python -m relgauge.blindobserverconnection 4 \
  --seed-mode flat_c2 --population 12 --generations 8 \
  --out example_results/seeded_observer_connection.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import observerboundarygeometry as OBG
    from . import observerframebundle as OBF
except Exception:  # pragma: no cover
    import observerboundarygeometry as OBG  # type: ignore
    import observerframebundle as OBF  # type: ignore


@dataclass
class Candidate:
    system: OBG.FiniteRelationalSystem
    score: float
    summary: Dict
    graph_id: int


def clone_system(sys: OBG.FiniteRelationalSystem) -> OBG.FiniteRelationalSystem:
    return OBG.FiniteRelationalSystem(q=int(sys.q), preds=[tuple(ps) for ps in sys.preds], tables=[np.array(t, copy=True) for t in sys.tables])


def randomize_tables(sys: OBG.FiniteRelationalSystem, rng: np.random.Generator) -> OBG.FiniteRelationalSystem:
    out = clone_system(sys)
    out.tables = [rng.integers(0, out.q, size=len(t), dtype=np.int16) for t in out.tables]
    return out


def mutate_system(
    sys: OBG.FiniteRelationalSystem,
    rng: np.random.Generator,
    mutation_rate: float = 0.05,
    min_mutations: int = 1,
    max_mutations: Optional[int] = None,
) -> OBG.FiniteRelationalSystem:
    """Mutate only local rule tables; graph/topology remains fixed."""
    out = clone_system(sys)
    table_sizes = [len(t) for t in out.tables]
    total_entries = int(sum(table_sizes))
    if total_entries <= 0:
        return out
    if max_mutations is None or max_mutations <= 0:
        n_mut = int(rng.binomial(total_entries, min(1.0, max(0.0, mutation_rate))))
        n_mut = max(int(min_mutations), n_mut)
    else:
        n_mut = int(rng.integers(int(min_mutations), int(max_mutations) + 1))
    # Draw table-entry positions with replacement; repeated hits are harmless.
    for _ in range(max(0, n_mut)):
        flat = int(rng.integers(0, total_entries))
        acc = 0
        for vi, size in enumerate(table_sizes):
            if flat < acc + size:
                idx = flat - acc
                old = int(out.tables[vi][idx])
                if out.q <= 1:
                    new = old
                else:
                    new = int(rng.integers(0, out.q - 1))
                    if new >= old:
                        new += 1
                out.tables[vi][idx] = new
                break
            acc += size
    return out


def make_initial_system(
    q: int,
    vertices: int,
    graph_mode: str,
    components: int,
    edge_prob: float,
    inter_prob: float,
    extra_intra_prob: float,
    max_pred: Optional[int],
    seed_mode: str,
    rng: np.random.Generator,
) -> OBG.FiniteRelationalSystem:
    seed = seed_mode.lower()
    if seed in {"flat_c2", "flat"}:
        return OBF.make_frame_diamond_control(q=q, twist=False)
    if seed in {"twist_c2", "twist"}:
        return OBF.make_frame_diamond_control(q=q, twist=True)
    if seed in {"theta_flat_c2", "flat_theta"}:
        return OBF.make_frame_theta_control(q=q, twists=(False, False, False))
    if seed in {"theta_twist_c2", "twist_theta"}:
        return OBF.make_frame_theta_control(q=q, twists=(False, False, True))
    if seed in {"double_flat_c2", "flat_double"}:
        return OBF.make_frame_double_diamond_control(q=q, twists=(False, False))
    if seed in {"double_twist_c2", "twist_double"}:
        return OBF.make_frame_double_diamond_control(q=q, twists=(True, False))
    if seed in {"random", "blind"}:
        return OBF.make_system_for_mode(q, vertices, graph_mode, components, edge_prob, inter_prob, extra_intra_prob, max_pred, rng)
    if seed in {"randomized_flat_topology", "diamond_random_rules"}:
        return OBF.make_random_frame_diamond(q=q, rng=rng, max_pred=max_pred)
    if seed in {"theta_random_rules", "randomized_theta_topology"}:
        return OBF.make_random_frame_theta(q=q, rng=rng, max_pred=max_pred)
    if seed in {"double_random_rules", "randomized_double_topology"}:
        return OBF.make_random_frame_double_diamond(q=q, rng=rng, max_pred=max_pred)
    raise ValueError(f"unknown seed mode: {seed_mode}")


def score_from_summary(summary: Dict, cycle_objective: str = "best") -> float:
    """Pure consistency score; no flatness, group, or curvature target.

    ``cycle_objective='best'`` keeps the single-diamond v4 behavior: it rewards
    the best completed automorphism-valid loop.  ``cycle_objective='all'`` is
    for theta/two-diamond arenas: it rewards the fraction of discovered usable
    diamonds/independent chords that close as automorphisms.  Neither mode
    rewards identity/non-identity, C2, group order, flux, or matter.
    """
    mode = str(cycle_objective).lower()
    soft_edge = float(summary.get("mean_edge_mi_norm", 0.0))
    edge = float(summary.get("live_edge_quotient_fraction", 0.0))
    true_frame = float(summary.get("true_live_frame_fraction", summary.get("live_frame_fraction", 0.0)))
    true_transport = float(summary.get("true_live_frame_transport_fraction", summary.get("live_frame_transport_fraction", 0.0)))
    if mode == "all":
        branch = float(summary.get("valid_holonomy_fraction", 0.0))
        two_branch = float(summary.get("loop_automorphism_valid_fraction", 0.0))
        complete_branch = float(summary.get("valid_holonomy_fraction", 0.0))
        loop_auto = float(summary.get("loop_automorphism_valid_fraction", 0.0))
        if int(summary.get("global_chord_count", 0)) > 0:
            global_auto = float(int(summary.get("global_valid_holonomy", 0)) / max(1, int(summary.get("global_chord_count", 0))))
            loop_auto = max(loop_auto, global_auto)
    else:
        branch = float(summary.get("max_branch_completion", 0.0))
        two_branch = float(summary.get("max_two_branch_completion", 0.0))
        complete_branch = float(summary.get("max_complete_branch", 0.0))
        loop_auto = float(summary.get("max_loop_automorphism_validity", 0.0))
        if loop_auto <= 0.0 and int(summary.get("n_valid_holonomy", 0)) > 0:
            loop_auto = 1.0
        global_auto = 1.0 if int(summary.get("global_valid_holonomy", 0)) > 0 else 0.0
        loop_auto = max(loop_auto, global_auto)
    usable = 1.0 if int(summary.get("n_usable_diamonds", 0)) > 0 or int(summary.get("global_chord_count", 0)) > 0 else 0.0
    auto_valid = loop_auto * usable
    score = float(
        0.10 * soft_edge
        + 0.15 * edge
        + 0.15 * true_frame
        + 0.15 * true_transport
        + 0.15 * branch
        + 0.15 * two_branch
        + 0.05 * complete_branch
        + 0.10 * auto_valid
    )
    return float(max(0.0, min(1.0, score)))


def evaluate_candidate(
    sys: OBG.FiniteRelationalSystem,
    graph_id: int,
    rng: np.random.Generator,
    max_channel_inputs: int,
    max_channel_backgrounds: int,
    max_state_samples: int,
    state_warmup: int,
    cycle_max_len: int,
    min_frame_ports: int,
    frame_coordinate_mode: str,
    cycle_objective: str = "best",
) -> Candidate:
    _fr, _er, _tr, _cr, summary = OBF.analyze_system_bundle(
        sys,
        graph_id=graph_id,
        max_channel_inputs=max_channel_inputs,
        max_channel_backgrounds=max_channel_backgrounds,
        max_state_samples=max_state_samples,
        state_warmup=state_warmup,
        cycle_max_len=cycle_max_len,
        min_frame_ports=min_frame_ports,
        frame_coordinate_mode=frame_coordinate_mode,
        rng=rng,
    )
    score = score_from_summary(summary, cycle_objective=cycle_objective)
    return Candidate(system=sys, score=score, summary=summary, graph_id=graph_id)


def evolve_observer_connection(
    q: int = 4,
    graph_mode: str = "frame_random_diamond",
    vertices: int = 8,
    components: int = 5,
    edge_prob: float = 0.16,
    inter_prob: float = 0.35,
    extra_intra_prob: float = 0.25,
    max_pred: Optional[int] = 4,
    seed_mode: str = "random",
    population: int = 24,
    generations: int = 30,
    elite: int = 4,
    mutation_rate: float = 0.06,
    min_mutations: int = 1,
    max_mutations: Optional[int] = None,
    random_injection: float = 0.10,
    max_channel_inputs: int = 128,
    max_channel_backgrounds: int = 128,
    max_state_samples: int = 1024,
    state_warmup: int = 4,
    cycle_max_len: int = 6,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    cycle_objective: str = "best",
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[object, List[Candidate], Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for evolve_observer_connection")
    rng0 = np.random.default_rng(seed)
    pop: List[OBG.FiniteRelationalSystem] = []
    for _ in range(int(population)):
        grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
        sys = make_initial_system(q, vertices, graph_mode, components, edge_prob, inter_prob, extra_intra_prob, max_pred, seed_mode, grng)
        if seed_mode.lower() in {"flat_c2", "flat", "twist_c2", "twist"}:
            # Make seeded recovery nontrivial: most individuals are lightly mutated.
            if len(pop) > 0:
                sys = mutate_system(sys, grng, mutation_rate=mutation_rate, min_mutations=min_mutations, max_mutations=max_mutations)
        pop.append(sys)

    history: List[Dict] = []
    best_ever: Optional[Candidate] = None
    final_candidates: List[Candidate] = []

    for gen in range(int(generations) + 1):
        evaluated: List[Candidate] = []
        for i, sys in enumerate(pop):
            erng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
            cand = evaluate_candidate(
                sys,
                graph_id=gen * max(1, int(population)) + i,
                rng=erng,
                max_channel_inputs=max_channel_inputs,
                max_channel_backgrounds=max_channel_backgrounds,
                max_state_samples=max_state_samples,
                state_warmup=state_warmup,
                cycle_max_len=cycle_max_len,
                min_frame_ports=min_frame_ports,
                frame_coordinate_mode=frame_coordinate_mode,
                cycle_objective=cycle_objective,
            )
            evaluated.append(cand)
        evaluated.sort(key=lambda c: c.score, reverse=True)
        if best_ever is None or evaluated[0].score > best_ever.score:
            best_ever = evaluated[0]
        final_candidates = evaluated
        best = evaluated[0]
        row = dict(
            generation=int(gen),
            best_score=float(best.score),
            mean_score=float(np.mean([c.score for c in evaluated])),
            median_score=float(np.median([c.score for c in evaluated])),
            best_mean_edge_mi_norm=float(best.summary.get("mean_edge_mi_norm", 0.0)),
            best_live_edge_quotient_fraction=float(best.summary.get("live_edge_quotient_fraction", 0.0)),
            best_port_label_frame_fraction=float(best.summary.get("port_label_frame_fraction", 0.0)),
            best_single_port_label_frame_fraction=float(best.summary.get("single_port_label_frame_fraction", 0.0)),
            best_live_frame_fraction=float(best.summary.get("live_frame_fraction", 0.0)),
            best_true_live_frame_fraction=float(best.summary.get("true_live_frame_fraction", best.summary.get("live_frame_fraction", 0.0))),
            best_port_label_transport_fraction=float(best.summary.get("port_label_transport_fraction", 0.0)),
            best_live_frame_transport_fraction=float(best.summary.get("live_frame_transport_fraction", 0.0)),
            best_true_live_frame_transport_fraction=float(best.summary.get("true_live_frame_transport_fraction", best.summary.get("live_frame_transport_fraction", 0.0))),
            best_max_branch_completion=float(best.summary.get("max_branch_completion", 0.0)),
            best_max_two_branch_completion=float(best.summary.get("max_two_branch_completion", 0.0)),
            best_max_complete_branch=float(best.summary.get("max_complete_branch", 0.0)),
            best_max_two_complete_branches=float(best.summary.get("max_two_complete_branches", 0.0)),
            best_max_path_agreement=float(best.summary.get("max_path_agreement", 0.0)),
            best_max_path_comparability=float(best.summary.get("max_path_comparability", 0.0)),
            best_max_loop_automorphism_validity=float(best.summary.get("max_loop_automorphism_validity", 0.0)),
            best_valid_connections=int(best.summary.get("n_valid_connections", 0)),
            best_flat_connections=int(best.summary.get("n_flat_connections", best.summary.get("n_valid_connections", 0))),
            best_loop_automorphism_valid=int(best.summary.get("n_loop_automorphism_valid", best.summary.get("n_valid_holonomy", 0))),
            best_valid_holonomy=int(best.summary.get("n_valid_holonomy", 0)),
            best_nontrivial_holonomy=int(best.summary.get("n_nontrivial_holonomy", 0)),
            best_global_valid_holonomy=int(best.summary.get("global_valid_holonomy", 0)),
            best_global_nontrivial_holonomy=int(best.summary.get("global_nontrivial_holonomy", 0)),
            best_group_hint="posthoc",
        )
        history.append(row)
        if verbose:
            print(
                f"observer-connection gen={gen} best={row['best_score']:.4f} "
                f"mi={row['best_mean_edge_mi_norm']:.3f} "
                f"edge={row['best_live_edge_quotient_fraction']:.3f} "
                f"portframe={row['best_port_label_frame_fraction']:.3f} "
                f"frame={row['best_true_live_frame_fraction']:.3f} "
                f"transport={row['best_true_live_frame_transport_fraction']:.3f} "
                f"branch={row['best_max_branch_completion']:.3f}/{row['best_max_two_branch_completion']:.3f} "
                f"cmp={row['best_max_path_comparability']:.3f} "
                f"auto={row['best_max_loop_automorphism_validity']:.3f} "
                f"flat={row['best_flat_connections']} hol={row['best_valid_holonomy']} global={row['best_global_valid_holonomy']} nontriv={row['best_nontrivial_holonomy']}",
                flush=True,
            )
        if gen >= int(generations):
            break
        # Elitist mutation/recombination-free search.  The only pressure is the
        # pure consistency score above.
        parents = evaluated[: max(1, min(int(elite), len(evaluated)))]
        new_pop: List[OBG.FiniteRelationalSystem] = [clone_system(c.system) for c in parents]
        while len(new_pop) < int(population):
            grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
            if grng.random() < random_injection:
                child = make_initial_system(q, vertices, graph_mode, components, edge_prob, inter_prob, extra_intra_prob, max_pred, "random", grng)
            else:
                parent = parents[int(grng.integers(0, len(parents)))]
                child = mutate_system(parent.system, grng, mutation_rate=mutation_rate, min_mutations=min_mutations, max_mutations=max_mutations)
            new_pop.append(child)
        pop = new_pop

    hist_df = pd.DataFrame(history)
    assert best_ever is not None
    final_best = final_candidates[0]
    summary = dict(
        verdict=(
            "BLIND OBSERVER CONNECTION FOUND NONTRIVIAL LOOP AUTOMORPHISM"
            if (int(final_best.summary.get("n_nontrivial_holonomy", 0)) > 0 or int(final_best.summary.get("global_nontrivial_holonomy", 0)) > 0)
            else (
                "BLIND OBSERVER CONNECTION FOUND AUTOMORPHISM-VALID FRAME TRANSPORT"
                if (int(final_best.summary.get("n_loop_automorphism_valid", final_best.summary.get("n_valid_holonomy", 0))) > 0 or int(final_best.summary.get("global_valid_holonomy", 0)) > 0)
                else "NO BLIND OBSERVER CONNECTION FOUND in this run"
            )
        ),
        q=int(q),
        graph_mode=graph_mode,
        seed_mode=seed_mode,
        population=int(population),
        generations=int(generations),
        min_frame_ports=int(min_frame_ports),
        frame_coordinate_mode=str(frame_coordinate_mode),
        cycle_objective=str(cycle_objective),
        final_best_score=float(final_best.score),
        all_time_best_score=float(best_ever.score),
        final_best_summary=final_best.summary,
        all_time_best_summary=best_ever.summary,
        final_found_connection=bool(int(final_best.summary.get("n_loop_automorphism_valid", final_best.summary.get("n_valid_holonomy", 0))) > 0 or int(final_best.summary.get("global_valid_holonomy", 0)) > 0),
        final_found_flat_connection=bool(int(final_best.summary.get("n_flat_connections", final_best.summary.get("n_valid_connections", 0))) > 0),
        final_found_automorphism_valid_connection=bool(int(final_best.summary.get("n_loop_automorphism_valid", final_best.summary.get("n_valid_holonomy", 0))) > 0 or int(final_best.summary.get("global_valid_holonomy", 0)) > 0),
        final_found_valid_holonomy=bool(int(final_best.summary.get("n_valid_holonomy", 0)) > 0),
        final_found_nontrivial_holonomy=bool(int(final_best.summary.get("n_nontrivial_holonomy", 0)) > 0),
        final_found_global_holonomy=bool(int(final_best.summary.get("global_valid_holonomy", 0)) > 0),
        final_found_global_nontrivial_holonomy=bool(int(final_best.summary.get("global_nontrivial_holonomy", 0)) > 0),
        final_branch_completion=float(final_best.summary.get("max_branch_completion", 0.0)),
        final_two_branch_completion=float(final_best.summary.get("max_two_branch_completion", 0.0)),
        final_loop_automorphism_validity=float(final_best.summary.get("max_loop_automorphism_validity", 0.0)),
        final_path_agreement=float(final_best.summary.get("max_path_agreement", 0.0)),
        all_time_found_connection=bool(int(best_ever.summary.get("n_loop_automorphism_valid", best_ever.summary.get("n_valid_holonomy", 0))) > 0 or int(best_ever.summary.get("global_valid_holonomy", 0)) > 0),
        all_time_found_flat_connection=bool(int(best_ever.summary.get("n_flat_connections", best_ever.summary.get("n_valid_connections", 0))) > 0),
        all_time_found_automorphism_valid_connection=bool(int(best_ever.summary.get("n_loop_automorphism_valid", best_ever.summary.get("n_valid_holonomy", 0))) > 0 or int(best_ever.summary.get("global_valid_holonomy", 0)) > 0),
        all_time_found_valid_holonomy=bool(int(best_ever.summary.get("n_valid_holonomy", 0)) > 0),
        all_time_found_nontrivial_holonomy=bool(int(best_ever.summary.get("n_nontrivial_holonomy", 0)) > 0),
        all_time_found_global_holonomy=bool(int(best_ever.summary.get("global_valid_holonomy", 0)) > 0),
        all_time_found_global_nontrivial_holonomy=bool(int(best_ever.summary.get("global_nontrivial_holonomy", 0)) > 0),
        all_time_branch_completion=float(best_ever.summary.get("max_branch_completion", 0.0)),
        all_time_two_branch_completion=float(best_ever.summary.get("max_two_branch_completion", 0.0)),
        all_time_loop_automorphism_validity=float(best_ever.summary.get("max_loop_automorphism_validity", 0.0)),
        all_time_path_agreement=float(best_ever.summary.get("max_path_agreement", 0.0)),
    )
    return hist_df, [best_ever, final_best] + final_candidates[: max(1, min(10, len(final_candidates)))], summary


def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def write_outputs(hist_df, winners: List[Candidate], summary: Dict, out: str, winners_path: Optional[str]) -> None:
    hist_df.to_csv(out, index=False)
    with open(derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if winners_path:
        payload = []
        seen = set()
        for c in winners:
            key = id(c.system)
            if key in seen:
                continue
            seen.add(key)
            payload.append(dict(system=c.system, score=c.score, summary=c.summary, graph_id=c.graph_id))
        with open(winners_path, "wb") as f:
            pickle.dump(payload, f)


def make_plot(hist_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(hist_df):
        ax.plot(hist_df["generation"], hist_df["best_score"], label="best")
        ax.plot(hist_df["generation"], hist_df["mean_score"], label="mean")
    ax.set_title(summary.get("verdict", "blind observer connection"))
    ax.set_xlabel("generation")
    ax.set_ylabel("automorphism-valid consistency score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blind observer-relative connection selection")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--graph-mode", choices=["frame_random_diamond", "componented", "er", "frame_flat_c2", "frame_twist_c2", "frame_theta_flat_c2", "frame_theta_twist_c2", "frame_random_theta", "frame_double_flat_c2", "frame_double_twist_c2", "frame_random_double_diamond"], default="frame_random_diamond")
    p.add_argument("--vertices", type=int, default=8)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.16)
    p.add_argument("--inter-prob", type=float, default=0.35)
    p.add_argument("--extra-intra-prob", type=float, default=0.25)
    p.add_argument("--max-pred", type=int, default=4)
    p.add_argument("--seed-mode", choices=["random", "flat_c2", "twist_c2", "randomized_flat_topology", "diamond_random_rules", "theta_flat_c2", "theta_twist_c2", "theta_random_rules", "double_flat_c2", "double_twist_c2", "double_random_rules"], default="random")
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--mutation-rate", type=float, default=0.06)
    p.add_argument("--min-mutations", type=int, default=1)
    p.add_argument("--max-mutations", type=int, default=0)
    p.add_argument("--random-injection", type=float, default=0.10)
    p.add_argument("--max-channel-inputs", type=int, default=128)
    p.add_argument("--max-channel-backgrounds", type=int, default=128)
    p.add_argument("--max-state-samples", type=int, default=1024)
    p.add_argument("--state-warmup", type=int, default=4)
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--min-frame-ports", type=int, default=2, help="minimum incident ports for a true observer frame")
    p.add_argument("--frame-coordinate-mode", choices=["local_charts", "common_factor"], default="local_charts",
                   help="local_charts keeps oriented endpoint transition maps; common_factor uses shared equalizer coordinates")
    p.add_argument("--cycle-objective", choices=["best", "all"], default="best",
                   help="best rewards one completed loop; all rewards the fraction of loops/chords closed as automorphisms")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/blind_observer_connection.csv")
    p.add_argument("--winners", default="example_results/blind_observer_connection_winners.pkl")
    p.add_argument("--plot", default="example_results/fig_blind_observer_connection.png")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    max_pred = None if args.max_pred is None or args.max_pred <= 0 else int(args.max_pred)
    max_mut = None if args.max_mutations is None or args.max_mutations <= 0 else int(args.max_mutations)
    hist_df, winners, summary = evolve_observer_connection(
        q=args.q,
        graph_mode=args.graph_mode,
        vertices=args.vertices,
        components=args.components,
        edge_prob=args.edge_prob,
        inter_prob=args.inter_prob,
        extra_intra_prob=args.extra_intra_prob,
        max_pred=max_pred,
        seed_mode=args.seed_mode,
        population=args.population,
        generations=args.generations,
        elite=args.elite,
        mutation_rate=args.mutation_rate,
        min_mutations=args.min_mutations,
        max_mutations=max_mut,
        random_injection=args.random_injection,
        max_channel_inputs=args.max_channel_inputs,
        max_channel_backgrounds=args.max_channel_backgrounds,
        max_state_samples=args.max_state_samples,
        state_warmup=args.state_warmup,
        cycle_max_len=args.cycle_max_len,
        min_frame_ports=args.min_frame_ports,
        frame_coordinate_mode=args.frame_coordinate_mode,
        cycle_objective=args.cycle_objective,
        seed=args.seed,
        verbose=not args.quiet,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.winners:
        os.makedirs(os.path.dirname(args.winners) or ".", exist_ok=True)
    write_outputs(hist_df, winners, summary, args.out, args.winners)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plot(hist_df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {derived_path(args.out, '_summary').replace('.csv', '.json')}")
    if args.winners:
        print(f"wrote {args.winners}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
