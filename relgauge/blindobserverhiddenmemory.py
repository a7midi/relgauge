"""
blindobserverhiddenmemory.py

Selection experiment for hidden observer-memory schedule absorption.

This is the selection companion to observerhiddenmemory.py.  It preserves the
observer-frame connection arena and selects only for visible dynamics becoming
schedule-consistent when conditioned on a compressed quotient of observer-internal
memory.

The score does NOT reward motion, nontrivial recurrent memory quotients, C2,
nontrivial holonomy, flux, particles, localization, or lifetime.  It rewards:

    - preserving the selected observer-frame connection gate;
    - hidden-conditioned visible transition determinism;
    - reduction of visible schedule ambiguity by hidden memory;
    - existence of a compressed memory quotient beyond visible state alone;
    - deterministic color-preserving memory quotient transitions.

If recurrent/moving dynamics appears, it is reported post hoc.
"""
from __future__ import annotations

import argparse
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
    from . import blindobserverconnection as BOC
    from . import observerhiddenmemory as OHM
except Exception:  # pragma: no cover
    import observerboundarygeometry as OBG  # type: ignore
    import blindobserverconnection as BOC  # type: ignore
    import observerhiddenmemory as OHM  # type: ignore


@dataclass
class HiddenMemoryCandidate:
    system: OBG.FiniteRelationalSystem
    score: float
    connection_score: float
    memory_score: float
    summary: Dict
    graph_id: int


def clone_system(sys: OBG.FiniteRelationalSystem) -> OBG.FiniteRelationalSystem:
    return BOC.clone_system(sys)


def load_system_from_winners(path: str, index: int = 0) -> OBG.FiniteRelationalSystem:
    return OHM.load_system_from_winners(path, index)


def memory_consistency_score(summary: Dict, connection_weight: float = 0.35, gate_fail_scale: float = 0.49) -> Tuple[float, float, float]:
    bundle_summary = dict(summary.get("bundle_summary", {}) or {})
    conn = BOC.score_from_summary(bundle_summary, cycle_objective="all")
    gate_passed = bool(summary.get("connection_gate_passed", False))
    gate_progress = float(summary.get("connection_gate_progress", 0.0))

    if not gate_passed:
        total = float(gate_fail_scale) * float(gate_progress) * max(float(conn), 0.0)
        return total, float(conn), 0.0

    visible_acc = float(summary.get("visible_only_transition_determinism_accuracy", 0.0))
    hidden_acc = float(summary.get("hidden_conditioned_visible_determinism_accuracy", 0.0))
    ambiguity_reduction = float(summary.get("hidden_memory_ambiguity_reduction", 0.0))
    q_det = float(summary.get("memory_quotient_source_block_deterministic_fraction", 0.0))
    conflict_free = 1.0 - float(summary.get("memory_quotient_visible_conflict_source_fraction", 1.0))
    extra_fraction = float(summary.get("memory_extra_fraction", 0.0))
    compression_ratio = float(summary.get("memory_compression_ratio", 1.0))
    # Prefer nontrivial but compressed hidden memory: beyond visible-only, below full joint.
    compression_score = max(0.0, min(1.0, 1.0 - compression_ratio)) if extra_fraction > 0 else 0.0
    hidden_gain = max(0.0, hidden_acc - visible_acc)

    # V2 visible-quotienting terms.  These reward only schedule-consistent
    # quotient structure and retained/compressed distinctions, not motion,
    # recurrence, holonomy, C2, or any target physical label.
    vq_det = float(summary.get("visible_schedule_quotient_source_block_deterministic_fraction", 0.0))
    vq_ret = float(summary.get("visible_schedule_quotient_retention_fraction", 0.0))
    jq_det = float(summary.get("joint_schedule_quotient_source_block_deterministic_fraction", 0.0))
    jq_ret = float(summary.get("joint_schedule_quotient_retention_fraction", 0.0))
    jq_comp = float(summary.get("joint_schedule_quotient_compression_fraction", 0.0))
    iv_ret = float(summary.get("induced_visible_quotient_retention_fraction", 0.0))
    iv_det = float(summary.get("induced_visible_quotient_transition_determinism_accuracy", 0.0))
    vq_classes = int(summary.get("visible_schedule_quotient_classes", 0))
    jq_classes = int(summary.get("joint_schedule_quotient_classes", 0))
    distinct_visible = int(summary.get("distinct_visible_states", 0))
    distinct_joint = int(summary.get("distinct_joint_states", 0))
    vq_noncollapsed = 1.0 if 1 < vq_classes < max(2, distinct_visible) else 0.0
    jq_noncollapsed = 1.0 if 1 < jq_classes < max(2, distinct_joint) else 0.0
    visible_quotient_score = (
        0.20 * vq_det
        + 0.15 * vq_ret
        + 0.10 * vq_noncollapsed
        + 0.20 * jq_det
        + 0.15 * jq_comp
        + 0.10 * jq_noncollapsed
        + 0.05 * iv_ret
        + 0.05 * iv_det
    )

    memory_score = (
        0.20 * hidden_acc
        + 0.15 * ambiguity_reduction
        + 0.15 * q_det
        + 0.05 * conflict_free
        + 0.05 * min(1.0, extra_fraction)
        + 0.05 * compression_score
        + 0.05 * min(1.0, hidden_gain / 0.25)
        + 0.30 * visible_quotient_score
    )
    total = float(connection_weight) * float(conn) + (1.0 - float(connection_weight)) * float(memory_score)
    return float(total), float(conn), float(memory_score)


def evaluate_candidate(
    sys: OBG.FiniteRelationalSystem,
    graph_id: int,
    rng: np.random.Generator,
    q: int = 4,
    graph_mode: str = "frame_random_theta",
    max_channel_inputs: int = 128,
    max_channel_backgrounds: int = 128,
    max_state_samples: int = 1024,
    state_warmup: int = 0,
    schedule_samples: int = 6,
    cycle_max_len: int = 6,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    connection_weight: float = 0.35,
    gate_fail_scale: float = 0.49,
) -> HiddenMemoryCandidate:
    _p, _t, _q, _o, summary = OHM.run_hidden_memory_audit(
        sys=sys,
        q=q,
        rng=rng,
        graph_mode=graph_mode,
        max_channel_inputs=max_channel_inputs,
        max_channel_backgrounds=max_channel_backgrounds,
        max_state_samples=max_state_samples,
        state_warmup=state_warmup,
        schedule_samples=schedule_samples,
        cycle_max_len=cycle_max_len,
        min_frame_ports=min_frame_ports,
        frame_coordinate_mode=frame_coordinate_mode,
        connection_gate="auto",
    )
    score, conn, mem = memory_consistency_score(summary, connection_weight=connection_weight, gate_fail_scale=gate_fail_scale)
    return HiddenMemoryCandidate(system=clone_system(sys), score=score, connection_score=conn, memory_score=mem, summary=summary, graph_id=int(graph_id))


def make_initial_population(
    q: int,
    population: int,
    graph_mode: str,
    vertices: int,
    components: int,
    edge_prob: float,
    inter_prob: float,
    extra_intra_prob: float,
    max_pred: Optional[int],
    seed_mode: str,
    seed_winners: str,
    seed_winner_index: int,
    mutation_rate: float,
    min_mutations: int,
    max_mutations: Optional[int],
    rng0: np.random.Generator,
) -> List[OBG.FiniteRelationalSystem]:
    pop: List[OBG.FiniteRelationalSystem] = []
    if seed_winners:
        base = load_system_from_winners(seed_winners, seed_winner_index)
        pop.append(clone_system(base))
        while len(pop) < int(population):
            grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
            pop.append(BOC.mutate_system(base, grng, mutation_rate=mutation_rate, min_mutations=min_mutations, max_mutations=max_mutations))
        return pop
    for i in range(int(population)):
        grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
        sys = BOC.make_initial_system(
            q=q, vertices=vertices, graph_mode=graph_mode, components=components,
            edge_prob=edge_prob, inter_prob=inter_prob, extra_intra_prob=extra_intra_prob,
            max_pred=max_pred, seed_mode=seed_mode, rng=grng,
        )
        if i > 0 and str(seed_mode).lower() not in {"random", "blind", "theta_random_rules", "double_random_rules", "randomized_flat_topology", "diamond_random_rules", "randomized_theta_topology", "randomized_double_topology"}:
            sys = BOC.mutate_system(sys, grng, mutation_rate=mutation_rate, min_mutations=min_mutations, max_mutations=max_mutations)
        pop.append(sys)
    return pop


def evolve_observer_hidden_memory(
    q: int = 4,
    graph_mode: str = "frame_random_theta",
    vertices: int = 11,
    components: int = 5,
    edge_prob: float = 0.16,
    inter_prob: float = 0.35,
    extra_intra_prob: float = 0.25,
    max_pred: Optional[int] = 4,
    seed_mode: str = "theta_twist_c2",
    seed_winners: str = "",
    seed_winner_index: int = 0,
    population: int = 12,
    generations: int = 20,
    elite: int = 3,
    mutation_rate: float = 0.04,
    min_mutations: int = 1,
    max_mutations: Optional[int] = None,
    random_injection: float = 0.05,
    max_channel_inputs: int = 128,
    max_channel_backgrounds: int = 128,
    max_state_samples: int = 1024,
    state_warmup: int = 0,
    schedule_samples: int = 6,
    cycle_max_len: int = 6,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    connection_weight: float = 0.35,
    gate_fail_scale: float = 0.49,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[object, List[HiddenMemoryCandidate], Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for hidden-memory evolution")
    rng0 = np.random.default_rng(int(seed))
    pop = make_initial_population(
        q=q, population=population, graph_mode=graph_mode, vertices=vertices,
        components=components, edge_prob=edge_prob, inter_prob=inter_prob,
        extra_intra_prob=extra_intra_prob, max_pred=max_pred, seed_mode=seed_mode,
        seed_winners=seed_winners, seed_winner_index=seed_winner_index,
        mutation_rate=mutation_rate, min_mutations=min_mutations,
        max_mutations=max_mutations, rng0=rng0,
    )

    history: List[Dict] = []
    best_ever: Optional[HiddenMemoryCandidate] = None
    final_candidates: List[HiddenMemoryCandidate] = []

    for gen in range(int(generations) + 1):
        evaluated: List[HiddenMemoryCandidate] = []
        for i, sys in enumerate(pop):
            erng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
            cand = evaluate_candidate(
                sys=sys, graph_id=gen * max(1, int(population)) + i, rng=erng, q=q,
                graph_mode=graph_mode, max_channel_inputs=max_channel_inputs,
                max_channel_backgrounds=max_channel_backgrounds,
                max_state_samples=max_state_samples, state_warmup=state_warmup,
                schedule_samples=schedule_samples, cycle_max_len=cycle_max_len,
                min_frame_ports=min_frame_ports, frame_coordinate_mode=frame_coordinate_mode,
                connection_weight=connection_weight, gate_fail_scale=gate_fail_scale,
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
            best_connection_score=float(best.connection_score),
            best_memory_score=float(best.memory_score),
            best_connection_gate_passed=int(bool(best.summary.get("connection_gate_passed", False))),
            best_connection_gate_progress=float(best.summary.get("connection_gate_progress", 0.0)),
            best_visible_only_determinism=float(best.summary.get("visible_only_transition_determinism_accuracy", 0.0)),
            best_hidden_conditioned_visible_determinism=float(best.summary.get("hidden_conditioned_visible_determinism_accuracy", 0.0)),
            best_hidden_memory_ambiguity_reduction=float(best.summary.get("hidden_memory_ambiguity_reduction", 0.0)),
            best_memory_quotient_classes=int(best.summary.get("memory_quotient_classes", 0)),
            best_memory_quotient_source_block_deterministic_fraction=float(best.summary.get("memory_quotient_source_block_deterministic_fraction", 0.0)),
            best_memory_quotient_visible_conflict_source_fraction=float(best.summary.get("memory_quotient_visible_conflict_source_fraction", 0.0)),
            best_memory_quotient_recurrent_classes=int(best.summary.get("memory_quotient_recurrent_classes", 0)),
            best_memory_quotient_nontrivial_recurrent=int(bool(best.summary.get("memory_quotient_nontrivial_recurrent", False))),
            best_memory_quotient_visible_moving_edge_fraction=float(best.summary.get("memory_quotient_visible_moving_edge_fraction", 0.0)),
            best_visible_schedule_quotient_classes=int(best.summary.get("visible_schedule_quotient_classes", 0)),
            best_visible_schedule_quotient_retention_fraction=float(best.summary.get("visible_schedule_quotient_retention_fraction", 0.0)),
            best_visible_schedule_quotient_source_block_deterministic_fraction=float(best.summary.get("visible_schedule_quotient_source_block_deterministic_fraction", 0.0)),
            best_visible_schedule_quotient_recurrent_state_count=int(best.summary.get("visible_schedule_quotient_recurrent_state_count", 0)),
            best_visible_schedule_quotient_max_period=int(best.summary.get("visible_schedule_quotient_max_period", 0)),
            best_visible_schedule_quotient_nontrivial_periodic_recurrence=int(bool(best.summary.get("visible_schedule_quotient_nontrivial_periodic_recurrence", False))),
            best_joint_schedule_quotient_classes=int(best.summary.get("joint_schedule_quotient_classes", 0)),
            best_joint_schedule_quotient_retention_fraction=float(best.summary.get("joint_schedule_quotient_retention_fraction", 0.0)),
            best_joint_schedule_quotient_compression_fraction=float(best.summary.get("joint_schedule_quotient_compression_fraction", 0.0)),
            best_joint_schedule_quotient_source_block_deterministic_fraction=float(best.summary.get("joint_schedule_quotient_source_block_deterministic_fraction", 0.0)),
            best_induced_visible_quotient_classes=int(best.summary.get("induced_visible_quotient_classes", 0)),
            best_induced_visible_quotient_retention_fraction=float(best.summary.get("induced_visible_quotient_retention_fraction", 0.0)),
            best_induced_visible_quotient_max_period=int(best.summary.get("induced_visible_quotient_max_period", 0)),
            best_induced_visible_quotient_nontrivial_periodic_recurrence=int(bool(best.summary.get("induced_visible_quotient_nontrivial_periodic_recurrence", False))),
            best_visible_quotient_schedule_absorption_signal=int(bool(best.summary.get("visible_quotient_schedule_absorption_signal", False))),
            best_schedule_absorption_signal=int(bool(best.summary.get("schedule_absorption_signal", False))),
            best_dynamics_candidate=int(bool(best.summary.get("dynamics_candidate", False))),
            best_verdict=str(best.summary.get("verdict", "")),
        )
        history.append(row)
        if verbose:
            print(
                f"hidden-memory gen={gen} best={best.score:.4f} mem={best.memory_score:.4f} "
                f"vis={row['best_visible_only_determinism']:.3f} hid={row['best_hidden_conditioned_visible_determinism']:.3f} "
                f"qdet={row['best_memory_quotient_source_block_deterministic_fraction']:.3f}",
                flush=True,
            )
        if gen >= int(generations):
            break
        elites = evaluated[:max(1, int(elite))]
        new_pop = [clone_system(c.system) for c in elites]
        while len(new_pop) < int(population):
            if float(random_injection) > 0 and rng0.random() < float(random_injection):
                grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
                new_pop.append(BOC.make_initial_system(
                    q=q, vertices=vertices, graph_mode=graph_mode, components=components,
                    edge_prob=edge_prob, inter_prob=inter_prob, extra_intra_prob=extra_intra_prob,
                    max_pred=max_pred, seed_mode=seed_mode, rng=grng,
                ))
            else:
                parent = elites[int(rng0.integers(0, len(elites)))].system
                grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
                new_pop.append(BOC.mutate_system(parent, grng, mutation_rate=mutation_rate, min_mutations=min_mutations, max_mutations=max_mutations))
        pop = new_pop

    hist_df = pd.DataFrame(history)
    best = best_ever or final_candidates[0]
    final_best = final_candidates[0]
    winners = [
        dict(system=clone_system(best.system), summary=best.summary, score=float(best.score), kind="all_time_best"),
        dict(system=clone_system(final_best.system), summary=final_best.summary, score=float(final_best.score), kind="final_best"),
    ]

    final_absorb = bool(final_best.summary.get("schedule_absorption_signal", False))
    all_absorb = bool(best.summary.get("schedule_absorption_signal", False))
    final_vq_absorb = bool(final_best.summary.get("visible_quotient_schedule_absorption_signal", False))
    all_vq_absorb = bool(best.summary.get("visible_quotient_schedule_absorption_signal", False))
    final_dyn = bool(final_best.summary.get("dynamics_candidate", False))
    all_dyn = bool(best.summary.get("dynamics_candidate", False))
    if all_dyn or final_dyn:
        verdict = "BLIND VISIBLE-QUOTIENT MEMORY FOUND RECURRENT VISIBLE-DYNAMICS CANDIDATE"
    elif all_vq_absorb or final_vq_absorb:
        verdict = "BLIND VISIBLE-QUOTIENT MEMORY FOUND SCHEDULE-CONSISTENT OBSERVER QUOTIENT"
    elif all_absorb or final_absorb:
        verdict = "BLIND HIDDEN MEMORY FOUND SCHEDULE-ABSORBING OBSERVER MEMORY"
    elif bool(best.summary.get("hidden_memory_signal", False)):
        verdict = "BLIND HIDDEN MEMORY IMPROVED VISIBLE SCHEDULE CONSISTENCY"
    else:
        verdict = "BLIND HIDDEN MEMORY DID NOT FIND SCHEDULE-ABSORBING MEMORY"

    summary = dict(
        verdict=verdict,
        q=int(q), graph_mode=str(graph_mode), seed_mode=str(seed_mode),
        seed_winners=str(seed_winners), seed_winner_index=int(seed_winner_index),
        population=int(population), generations=int(generations),
        min_frame_ports=int(min_frame_ports), frame_coordinate_mode=str(frame_coordinate_mode),
        connection_weight=float(connection_weight),
        final_best_score=float(final_best.score), all_time_best_score=float(best.score),
        final_connection_score=float(final_best.connection_score), all_time_connection_score=float(best.connection_score),
        final_memory_score=float(final_best.memory_score), all_time_memory_score=float(best.memory_score),
        final_schedule_absorption_signal=final_absorb,
        all_time_schedule_absorption_signal=all_absorb,
        final_visible_quotient_schedule_absorption_signal=final_vq_absorb,
        all_time_visible_quotient_schedule_absorption_signal=all_vq_absorb,
        final_dynamics_candidate=final_dyn,
        all_time_dynamics_candidate=all_dyn,
        final_visible_schedule_quotient_max_period=int(final_best.summary.get("visible_schedule_quotient_max_period", 0)),
        all_time_visible_schedule_quotient_max_period=int(best.summary.get("visible_schedule_quotient_max_period", 0)),
        final_visible_schedule_quotient_nontrivial_periodic_recurrence=bool(final_best.summary.get("visible_schedule_quotient_nontrivial_periodic_recurrence", False)),
        all_time_visible_schedule_quotient_nontrivial_periodic_recurrence=bool(best.summary.get("visible_schedule_quotient_nontrivial_periodic_recurrence", False)),
        final_induced_visible_quotient_max_period=int(final_best.summary.get("induced_visible_quotient_max_period", 0)),
        all_time_induced_visible_quotient_max_period=int(best.summary.get("induced_visible_quotient_max_period", 0)),
        final_best_hidden_summary=final_best.summary,
        all_time_best_hidden_summary=best.summary,
    )
    return hist_df, winners, summary


def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def write_outputs(hist_df, winners, summary: Dict, out: str, winners_path: str) -> None:
    hist_df.to_csv(out, index=False)
    with open(derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if winners_path:
        with open(winners_path, "wb") as f:
            pickle.dump(winners, f)


def make_plot(hist_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(hist_df["generation"], hist_df["best_score"], label="best total")
    ax.plot(hist_df["generation"], hist_df["mean_score"], label="mean total")
    ax.plot(hist_df["generation"], hist_df["best_memory_score"], label="best memory")
    ax.set_xlabel("generation")
    ax.set_ylabel("hidden-memory consistency score")
    ax.set_title(summary.get("verdict", "blind hidden memory"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blind/seeded hidden observer-memory consistency selection")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--graph-mode", choices=[
        "frame_random_theta", "frame_theta_twist_c2", "frame_theta_flat_c2",
        "frame_random_diamond", "frame_twist_c2", "frame_flat_c2",
        "frame_random_double_diamond", "frame_double_twist_c2", "frame_double_flat_c2",
        "componented", "er"], default="frame_random_theta")
    p.add_argument("--vertices", type=int, default=11)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.16)
    p.add_argument("--inter-prob", type=float, default=0.35)
    p.add_argument("--extra-intra-prob", type=float, default=0.25)
    p.add_argument("--max-pred", type=int, default=4)
    p.add_argument("--seed-mode", default="theta_twist_c2")
    p.add_argument("--seed-winners", default="")
    p.add_argument("--seed-winner-index", type=int, default=0)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--elite", type=int, default=3)
    p.add_argument("--mutation-rate", type=float, default=0.04)
    p.add_argument("--min-mutations", type=int, default=1)
    p.add_argument("--max-mutations", type=int, default=0)
    p.add_argument("--random-injection", type=float, default=0.05)
    p.add_argument("--max-channel-inputs", type=int, default=128)
    p.add_argument("--max-channel-backgrounds", type=int, default=128)
    p.add_argument("--max-state-samples", type=int, default=1024)
    p.add_argument("--state-warmup", type=int, default=0)
    p.add_argument("--schedule-samples", type=int, default=6)
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--min-frame-ports", type=int, default=2)
    p.add_argument("--frame-coordinate-mode", choices=["local_charts", "common_factor"], default="local_charts")
    p.add_argument("--connection-weight", type=float, default=0.35)
    p.add_argument("--gate-fail-scale", type=float, default=0.49)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/blind_hidden_memory.csv")
    p.add_argument("--winners", default="example_results/blind_hidden_memory_winners.pkl")
    p.add_argument("--plot", default="example_results/fig_blind_hidden_memory.png")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    max_mut = None if int(args.max_mutations) <= 0 else int(args.max_mutations)
    max_pred = None if int(args.max_pred) <= 0 else int(args.max_pred)
    hist_df, winners, summary = evolve_observer_hidden_memory(
        q=int(args.q), graph_mode=str(args.graph_mode), vertices=int(args.vertices),
        components=int(args.components), edge_prob=float(args.edge_prob),
        inter_prob=float(args.inter_prob), extra_intra_prob=float(args.extra_intra_prob),
        max_pred=max_pred, seed_mode=str(args.seed_mode), seed_winners=str(args.seed_winners),
        seed_winner_index=int(args.seed_winner_index), population=int(args.population),
        generations=int(args.generations), elite=int(args.elite), mutation_rate=float(args.mutation_rate),
        min_mutations=int(args.min_mutations), max_mutations=max_mut,
        random_injection=float(args.random_injection), max_channel_inputs=int(args.max_channel_inputs),
        max_channel_backgrounds=int(args.max_channel_backgrounds), max_state_samples=int(args.max_state_samples),
        state_warmup=int(args.state_warmup), schedule_samples=int(args.schedule_samples),
        cycle_max_len=int(args.cycle_max_len), min_frame_ports=int(args.min_frame_ports),
        frame_coordinate_mode=str(args.frame_coordinate_mode), connection_weight=float(args.connection_weight),
        gate_fail_scale=float(args.gate_fail_scale), seed=int(args.seed), verbose=not bool(args.quiet),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.winners:
        os.makedirs(os.path.dirname(args.winners) or ".", exist_ok=True)
    write_outputs(hist_df, winners, summary, args.out, args.winners)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plot(hist_df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"final_best_hidden_summary", "all_time_best_hidden_summary"}}, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {derived_path(args.out, '_summary').replace('.csv', '.json')}")
    if args.winners:
        print(f"wrote {args.winners}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
