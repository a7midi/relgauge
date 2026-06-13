"""
blindobserversectiondynamics.py

Blind / seeded selection for observer-section residual consistency.

This module is the D-stage selection companion to observersectiondynamics.py.
The observer-frame bundle layer has already selected observer-relative frames and
oriented transports.  The section-dynamics audit then showed an important
negative: raw residual representatives can move, but coherent section/cohomology
residuals may fail closure under schedule freedom or collapse to a trivial
schedule quotient.

The purpose here is narrowly logical:

    select for closure and schedule-consistency of the *induced* coherent
    section-obstruction quotient.

The score does NOT reward motion, lifetime, localization, nontrivial recurrent
quotient, nontrivial holonomy, a target group, C2, flux, or particles.  Those are
reported only post-hoc.  The score rewards only:

    - the already-pure observer-frame connection consistency score;
    - existence of coherent observer-frame sections;
    - closure of coherent/minimal/syndrome residual layers under schedules;
    - schedule-deterministic transition behavior of those residual layers.

This is intended to answer the next clean Endpoint-D question:

    Can consistency selection make the section-obstruction quotient autonomous,
    before we ask whether it is nontrivial, localized, or moving?

Typical seeded run from a selected theta winner:

python -m relgauge.blindobserversectiondynamics 4 ^
  --seed-winners example_results/blind_theta_connection_q4_winners.pkl ^
  --seed-winner-index 0 ^
  --population 24 --generations 60 --elite 4 ^
  --mutation-rate 0.04 --random-injection 0.05 ^
  --max-state-samples 2048 --schedule-samples 8 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/blind_section_dynamics_theta.csv ^
  --winners example_results/blind_section_dynamics_theta_winners.pkl
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
    from . import observersectiondynamics as OSD
except Exception:  # pragma: no cover
    import observerboundarygeometry as OBG  # type: ignore
    import blindobserverconnection as BOC  # type: ignore
    import observersectiondynamics as OSD  # type: ignore


@dataclass
class SectionCandidate:
    system: OBG.FiniteRelationalSystem
    score: float
    connection_score: float
    section_score: float
    summary: Dict
    graph_id: int


def clone_system(sys: OBG.FiniteRelationalSystem) -> OBG.FiniteRelationalSystem:
    return BOC.clone_system(sys)


def load_system_from_winners(path: str, index: int = 0) -> OBG.FiniteRelationalSystem:
    return OSD.load_system_from_winners(path, index)


def _layer(summary: Dict, name: str) -> Dict:
    layers = summary.get("layer_summaries", {}) or {}
    return dict(layers.get(name, {}) or {})


def default_connection_gate_targets(graph_mode: str) -> Dict[str, int]:
    """Conservative expected connection sizes for the controlled arenas.

    These are *not* physics targets.  They preserve the already-discovered
    C-stage observer-frame arena while the D-stage selector searches for
    autonomous residual/defect quotients inside it.  Without this gate, the
    optimizer can make residual dynamics autonomous by deleting loops and
    transports, which is the degenerate route seen in the first D-stage run.
    """
    gm = str(graph_mode).lower()
    if "theta" in gm:
        return dict(live_edge_quotients=6, true_frames=5, true_transports=6,
                    cycle_basis=2, global_chords=2, global_holonomy=2)
    if "double" in gm:
        return dict(live_edge_quotients=8, true_frames=7, true_transports=8,
                    cycle_basis=2, global_chords=2, global_holonomy=2)
    if "diamond" in gm or "flat_c2" in gm or "twist_c2" in gm:
        return dict(live_edge_quotients=4, true_frames=4, true_transports=4,
                    cycle_basis=1, global_chords=1, global_holonomy=1)
    return dict(live_edge_quotients=0, true_frames=0, true_transports=0,
                cycle_basis=0, global_chords=0, global_holonomy=0)


def _ratio(actual: float, target: float) -> float:
    if target <= 0:
        return 1.0
    return max(0.0, min(1.0, float(actual) / float(target)))


def connection_gate_status(
    summary: Dict,
    gate_min_live_edge_quotients: int = 0,
    gate_min_true_frames: int = 0,
    gate_min_true_transports: int = 0,
    gate_min_cycle_basis: int = 0,
    gate_min_global_chords: int = 0,
    gate_min_global_holonomy: int = 0,
    require_no_single_port_leakage: bool = True,
) -> Tuple[bool, float, Dict]:
    """Return (passed, progress, details) for D-stage connection preservation.

    The gate checks that the candidate still contains the full multi-cycle
    observer-frame connection.  It deliberately ignores nontrivial holonomy and
    group order: flat and non-flat sectors are both allowed as long as the
    connection arena survives.
    """
    bundle = dict(summary.get("bundle_summary", {}) or {})
    actual = dict(
        live_edge_quotients=int(bundle.get("n_live_edge_quotients", 0)),
        true_frames=int(bundle.get("n_true_live_frames", 0)),
        true_transports=int(bundle.get("n_true_live_frame_transports", 0)),
        cycle_basis=int(summary.get("cycle_basis_size", 0)),
        global_chords=int(summary.get("global_chord_count", bundle.get("global_chord_count", 0))),
        global_holonomy=int(summary.get("global_valid_holonomy", bundle.get("global_valid_holonomy", 0))),
        full_boundary_violation=int(bundle.get("full_boundary_violation", summary.get("full_boundary_violation", 0))),
        single_port_label_fraction=float(bundle.get("single_port_label_frame_fraction", 0.0)),
    )
    target = dict(
        live_edge_quotients=int(gate_min_live_edge_quotients),
        true_frames=int(gate_min_true_frames),
        true_transports=int(gate_min_true_transports),
        cycle_basis=int(gate_min_cycle_basis),
        global_chords=int(gate_min_global_chords),
        global_holonomy=int(gate_min_global_holonomy),
    )
    ratios = {k: _ratio(actual[k], v) for k, v in target.items() if int(v) > 0}
    if require_no_single_port_leakage:
        ratios["no_single_port_leakage"] = 1.0 if actual["single_port_label_fraction"] <= 1e-12 else 0.0
    ratios["no_full_boundary_violation"] = 1.0 if actual["full_boundary_violation"] == 0 else 0.0
    progress = float(np.mean(list(ratios.values()))) if ratios else 1.0
    passed = bool(progress >= 0.999999)
    details = dict(actual=actual, target=target, ratios=ratios, progress=progress, passed=passed)
    return passed, progress, details


def section_consistency_score(
    summary: Dict,
    connection_weight: float = 0.40,
    cycle_objective: str = "all",
    require_connection_gate: bool = False,
    gate_min_live_edge_quotients: int = 0,
    gate_min_true_frames: int = 0,
    gate_min_true_transports: int = 0,
    gate_min_cycle_basis: int = 0,
    gate_min_global_chords: int = 0,
    gate_min_global_holonomy: int = 0,
    gate_fail_scale: float = 0.49,
) -> Tuple[float, float, float]:
    """Purity-preserving D-stage score with optional hard connection gate.

    Returns (total, connection_component, section_component).  The section
    component intentionally does not reward nontriviality, motion, localization,
    lifetime, group order, or curvature.  It rewards only coherent-section
    existence plus closure/determinism of induced residual layers.  The new
    ``coherence_defect`` layer is observer-local and defined for every state;
    it is the primary D-stage candidate because edge/cohomology residuals on a
    fixed connection are often static representatives.

    If ``require_connection_gate`` is true, candidates that do not preserve the
    selected connection arena receive no D-stage section credit.  This prevents
    the degenerate solution in which residual autonomy is achieved by deleting
    cycles/transports.
    """
    conn = BOC.score_from_summary(summary.get("bundle_summary", {}), cycle_objective=cycle_objective)

    coherent_fraction = float(summary.get("coherent_state_fraction", 0.0))
    coherent_presence = min(1.0, coherent_fraction / 0.25)

    coh = _layer(summary, "coherent_edge")
    syn = _layer(summary, "cycle_syndrome")
    ms = _layer(summary, "minimal_support")
    defect = _layer(summary, "coherence_defect")

    coherent_closed = float(summary.get("coherent_transition_closed_fraction", coh.get("closed_transition_fraction", 0.0)))
    coherent_det = float(summary.get("coherent_transition_determinism_accuracy", coh.get("transition_determinism_accuracy", 0.0)))
    syndrome_closed = float(summary.get("cycle_syndrome_transition_closed_fraction", syn.get("closed_transition_fraction", 0.0)))
    syndrome_det = float(syn.get("transition_determinism_accuracy", 0.0))
    min_closed = float(summary.get("minimal_support_transition_closed_fraction", ms.get("closed_transition_fraction", 0.0)))
    min_det = float(ms.get("transition_determinism_accuracy", 0.0))
    defect_closed = float(summary.get("coherence_defect_transition_closed_fraction", defect.get("closed_transition_fraction", 0.0)))
    defect_det = float(summary.get("coherence_defect_transition_determinism_accuracy", defect.get("transition_determinism_accuracy", 0.0)))

    section = float(
        0.15 * coherent_presence
        + 0.30 * defect_closed
        + 0.30 * defect_det
        + 0.07 * coherent_closed
        + 0.06 * coherent_det
        + 0.04 * min_closed
        + 0.03 * min_det
        + 0.03 * syndrome_closed
        + 0.02 * syndrome_det
    )
    section = max(0.0, min(1.0, section))

    if require_connection_gate:
        gate_pass, gate_progress, _gate = connection_gate_status(
            summary,
            gate_min_live_edge_quotients=gate_min_live_edge_quotients,
            gate_min_true_frames=gate_min_true_frames,
            gate_min_true_transports=gate_min_true_transports,
            gate_min_cycle_basis=gate_min_cycle_basis,
            gate_min_global_chords=gate_min_global_chords,
            gate_min_global_holonomy=gate_min_global_holonomy,
        )
        if not gate_pass:
            # Hard gate: section autonomy cannot compensate for deleting the
            # connection.  The progress term keeps a gradient toward recovering
            # the arena, but all gate failures score below any passing candidate.
            total = max(0.0, min(float(gate_fail_scale), float(gate_fail_scale) * gate_progress))
            return float(total), float(conn), float(section)
        # Lexicographic-ish: after the connection is preserved, section
        # consistency dominates; a small connection term breaks ties without
        # selecting flatness/nonflatness.
        total = 0.50 + 0.40 * section + 0.10 * conn
        return float(max(0.0, min(1.0, total))), float(conn), float(section)

    cw = max(0.0, min(1.0, float(connection_weight)))
    total = cw * conn + (1.0 - cw) * section
    return float(max(0.0, min(1.0, total))), float(conn), float(section)

def evaluate_candidate(
    sys: OBG.FiniteRelationalSystem,
    graph_id: int,
    rng: np.random.Generator,
    max_channel_inputs: int,
    max_channel_backgrounds: int,
    max_state_samples: int,
    state_warmup: int,
    schedule_samples: int,
    cycle_max_len: int,
    min_frame_ports: int,
    frame_coordinate_mode: str,
    max_local_support: int,
    max_gauge_flips: int,
    connection_weight: float,
    cycle_objective: str,
    require_connection_gate: bool,
    gate_min_live_edge_quotients: int,
    gate_min_true_frames: int,
    gate_min_true_transports: int,
    gate_min_cycle_basis: int,
    gate_min_global_chords: int,
    gate_min_global_holonomy: int,
    gate_fail_scale: float,
) -> SectionCandidate:
    _pattern_df, _transition_df, _quotient_df, summary = OSD.run_section_dynamics(
        sys=sys,
        q=int(sys.q),
        rng=rng,
        max_channel_inputs=max_channel_inputs,
        max_channel_backgrounds=max_channel_backgrounds,
        max_state_samples=max_state_samples,
        state_warmup=state_warmup,
        schedule_samples=schedule_samples,
        cycle_max_len=cycle_max_len,
        min_frame_ports=min_frame_ports,
        frame_coordinate_mode=frame_coordinate_mode,
        max_local_support=max_local_support,
        max_gauge_flips=max_gauge_flips,
    )
    score, conn_score, section_score = section_consistency_score(
        summary,
        connection_weight=connection_weight,
        cycle_objective=cycle_objective,
        require_connection_gate=require_connection_gate,
        gate_min_live_edge_quotients=gate_min_live_edge_quotients,
        gate_min_true_frames=gate_min_true_frames,
        gate_min_true_transports=gate_min_true_transports,
        gate_min_cycle_basis=gate_min_cycle_basis,
        gate_min_global_chords=gate_min_global_chords,
        gate_min_global_holonomy=gate_min_global_holonomy,
        gate_fail_scale=gate_fail_scale,
    )
    # Store gate diagnostics in the summary so history/verdicts can distinguish
    # genuine D-stage progress from degenerate connection collapse.
    gp, gprog, gdetails = connection_gate_status(
        summary,
        gate_min_live_edge_quotients=gate_min_live_edge_quotients,
        gate_min_true_frames=gate_min_true_frames,
        gate_min_true_transports=gate_min_true_transports,
        gate_min_cycle_basis=gate_min_cycle_basis,
        gate_min_global_chords=gate_min_global_chords,
        gate_min_global_holonomy=gate_min_global_holonomy,
    )
    summary["connection_gate_required"] = bool(require_connection_gate)
    summary["connection_gate_passed"] = bool(gp) if require_connection_gate else True
    summary["connection_gate_progress"] = float(gprog)
    summary["connection_gate_details"] = gdetails
    return SectionCandidate(system=sys, score=score, connection_score=conn_score, section_score=section_score, summary=summary, graph_id=graph_id)


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
            q=q,
            vertices=vertices,
            graph_mode=graph_mode,
            components=components,
            edge_prob=edge_prob,
            inter_prob=inter_prob,
            extra_intra_prob=extra_intra_prob,
            max_pred=max_pred,
            seed_mode=seed_mode,
            rng=grng,
        )
        if i > 0 and str(seed_mode).lower() not in {"random", "blind", "theta_random_rules", "double_random_rules", "randomized_flat_topology", "diamond_random_rules", "randomized_theta_topology", "randomized_double_topology"}:
            sys = BOC.mutate_system(sys, grng, mutation_rate=mutation_rate, min_mutations=min_mutations, max_mutations=max_mutations)
        pop.append(sys)
    return pop


def evolve_observer_section_dynamics(
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
    max_local_support: int = 2,
    max_gauge_flips: int = 4096,
    connection_weight: float = 0.40,
    cycle_objective: str = "all",
    connection_gate: str = "auto",
    gate_min_live_edge_quotients: int = 0,
    gate_min_true_frames: int = 0,
    gate_min_true_transports: int = 0,
    gate_min_cycle_basis: int = 0,
    gate_min_global_chords: int = 0,
    gate_min_global_holonomy: int = 0,
    gate_fail_scale: float = 0.49,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[object, List[SectionCandidate], Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for evolve_observer_section_dynamics")
    rng0 = np.random.default_rng(int(seed))
    gate_mode = str(connection_gate).lower()
    require_connection_gate = gate_mode not in {"off", "none", "false", "0"}
    if gate_mode == "auto":
        auto_targets = default_connection_gate_targets(graph_mode)
        gate_min_live_edge_quotients = int(gate_min_live_edge_quotients or auto_targets["live_edge_quotients"])
        gate_min_true_frames = int(gate_min_true_frames or auto_targets["true_frames"])
        gate_min_true_transports = int(gate_min_true_transports or auto_targets["true_transports"])
        gate_min_cycle_basis = int(gate_min_cycle_basis or auto_targets["cycle_basis"])
        gate_min_global_chords = int(gate_min_global_chords or auto_targets["global_chords"])
        gate_min_global_holonomy = int(gate_min_global_holonomy or auto_targets["global_holonomy"])
    pop = make_initial_population(
        q=q,
        population=population,
        graph_mode=graph_mode,
        vertices=vertices,
        components=components,
        edge_prob=edge_prob,
        inter_prob=inter_prob,
        extra_intra_prob=extra_intra_prob,
        max_pred=max_pred,
        seed_mode=seed_mode,
        seed_winners=seed_winners,
        seed_winner_index=seed_winner_index,
        mutation_rate=mutation_rate,
        min_mutations=min_mutations,
        max_mutations=max_mutations,
        rng0=rng0,
    )

    history: List[Dict] = []
    best_ever: Optional[SectionCandidate] = None
    final_candidates: List[SectionCandidate] = []

    for gen in range(int(generations) + 1):
        evaluated: List[SectionCandidate] = []
        for i, sys in enumerate(pop):
            erng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
            cand = evaluate_candidate(
                sys=sys,
                graph_id=gen * max(1, int(population)) + i,
                rng=erng,
                max_channel_inputs=max_channel_inputs,
                max_channel_backgrounds=max_channel_backgrounds,
                max_state_samples=max_state_samples,
                state_warmup=state_warmup,
                schedule_samples=schedule_samples,
                cycle_max_len=cycle_max_len,
                min_frame_ports=min_frame_ports,
                frame_coordinate_mode=frame_coordinate_mode,
                max_local_support=max_local_support,
                max_gauge_flips=max_gauge_flips,
                connection_weight=connection_weight,
                cycle_objective=cycle_objective,
                require_connection_gate=require_connection_gate,
                gate_min_live_edge_quotients=int(gate_min_live_edge_quotients),
                gate_min_true_frames=int(gate_min_true_frames),
                gate_min_true_transports=int(gate_min_true_transports),
                gate_min_cycle_basis=int(gate_min_cycle_basis),
                gate_min_global_chords=int(gate_min_global_chords),
                gate_min_global_holonomy=int(gate_min_global_holonomy),
                gate_fail_scale=float(gate_fail_scale),
            )
            evaluated.append(cand)
        evaluated.sort(key=lambda c: c.score, reverse=True)
        if best_ever is None or evaluated[0].score > best_ever.score:
            best_ever = evaluated[0]
        final_candidates = evaluated
        best = evaluated[0]
        layers = best.summary.get("layer_summaries", {}) or {}
        min_layer = layers.get("minimal_support", {}) or {}
        syn_layer = layers.get("cycle_syndrome", {}) or {}
        defect_layer = layers.get("coherence_defect", {}) or {}
        row = dict(
            generation=int(gen),
            best_score=float(best.score),
            mean_score=float(np.mean([c.score for c in evaluated])),
            median_score=float(np.median([c.score for c in evaluated])),
            best_connection_score=float(best.connection_score),
            best_section_score=float(best.section_score),
            best_global_valid_holonomy=int(best.summary.get("global_valid_holonomy", 0)),
            best_global_nontrivial_holonomy=int(best.summary.get("global_nontrivial_holonomy", 0)),
            best_connection_gate_passed=int(bool(best.summary.get("connection_gate_passed", False))),
            best_connection_gate_progress=float(best.summary.get("connection_gate_progress", 0.0)),
            best_cycle_basis_size=int(best.summary.get("cycle_basis_size", 0)),
            best_live_transport_edges=int(best.summary.get("n_live_transport_edges", 0)),
            best_coherent_state_fraction=float(best.summary.get("coherent_state_fraction", 0.0)),
            best_coherent_transition_closed_fraction=float(best.summary.get("coherent_transition_closed_fraction", 0.0)),
            best_coherent_transition_determinism_accuracy=float(best.summary.get("coherent_transition_determinism_accuracy", 0.0)),
            best_cycle_syndrome_transition_closed_fraction=float(best.summary.get("cycle_syndrome_transition_closed_fraction", 0.0)),
            best_cycle_syndrome_transition_determinism_accuracy=float(syn_layer.get("transition_determinism_accuracy", 0.0)),
            best_minimal_support_transition_closed_fraction=float(best.summary.get("minimal_support_transition_closed_fraction", 0.0)),
            best_minimal_support_transition_determinism_accuracy=float(min_layer.get("transition_determinism_accuracy", 0.0)),
            best_minimal_support_patterns=int(best.summary.get("minimal_support_patterns", 0)),
            best_minimal_support_schedule_quotient_classes=int(best.summary.get("minimal_support_schedule_quotient_classes", 0)),
            best_minimal_support_recurrent_quotient_classes=int(best.summary.get("minimal_support_recurrent_quotient_classes", 0)),
            best_minimal_support_nontrivial_recurrent_quotient=int(bool(best.summary.get("minimal_support_nontrivial_recurrent_quotient", False))),
            best_minimal_support_moving_transition_fraction=float(best.summary.get("minimal_support_moving_transition_fraction", 0.0)),
            best_coherence_defect_transition_closed_fraction=float(best.summary.get("coherence_defect_transition_closed_fraction", defect_layer.get("closed_transition_fraction", 0.0))),
            best_coherence_defect_transition_determinism_accuracy=float(best.summary.get("coherence_defect_transition_determinism_accuracy", defect_layer.get("transition_determinism_accuracy", 0.0))),
            best_coherence_defect_schedule_quotient_classes=int(best.summary.get("coherence_defect_schedule_quotient_classes", defect_layer.get("n_schedule_quotient_classes", 0))),
            best_coherence_defect_recurrent_quotient_classes=int(best.summary.get("coherence_defect_recurrent_quotient_classes", defect_layer.get("n_recurrent_quotient_classes", 0))),
            best_coherence_defect_nontrivial_recurrent_quotient=int(bool(best.summary.get("coherence_defect_nontrivial_recurrent_quotient", False))),
            best_coherence_defect_moving_transition_fraction=float(best.summary.get("coherence_defect_moving_transition_fraction", defect_layer.get("moving_transition_fraction", 0.0))),
            best_verdict=str(best.summary.get("verdict", "")),
        )
        history.append(row)
        if verbose:
            print(
                f"section-dynamics gen={gen} best={row['best_score']:.4f} "
                f"conn={row['best_connection_score']:.3f} sec={row['best_section_score']:.3f} "
                f"gate={row['best_connection_gate_passed']}:{row['best_connection_gate_progress']:.2f} "
                f"coh={row['best_coherent_state_fraction']:.3f} "
                f"def_closed={row['best_coherence_defect_transition_closed_fraction']:.3f} "
                f"def_det={row['best_coherence_defect_transition_determinism_accuracy']:.3f} "
                f"q={row['best_minimal_support_schedule_quotient_classes']} "
                f"rec={row['best_minimal_support_recurrent_quotient_classes']} "
                f"nontriv_rec={row['best_minimal_support_nontrivial_recurrent_quotient']} "
                f"hol={row['best_global_valid_holonomy']} nontriv_hol={row['best_global_nontrivial_holonomy']}",
                flush=True,
            )
        if gen >= int(generations):
            break
        parents = evaluated[: max(1, min(int(elite), len(evaluated)))]
        new_pop: List[OBG.FiniteRelationalSystem] = [clone_system(c.system) for c in parents]
        while len(new_pop) < int(population):
            grng = np.random.default_rng(int(rng0.integers(0, 2**32 - 1)))
            if grng.random() < float(random_injection):
                child = BOC.make_initial_system(
                    q=q,
                    vertices=vertices,
                    graph_mode=graph_mode,
                    components=components,
                    edge_prob=edge_prob,
                    inter_prob=inter_prob,
                    extra_intra_prob=extra_intra_prob,
                    max_pred=max_pred,
                    seed_mode="random",
                    rng=grng,
                )
            else:
                parent = parents[int(grng.integers(0, len(parents)))]
                child = BOC.mutate_system(parent.system, grng, mutation_rate=mutation_rate, min_mutations=min_mutations, max_mutations=max_mutations)
            new_pop.append(child)
        pop = new_pop

    hist_df = pd.DataFrame(history)
    assert best_ever is not None
    final_best = final_candidates[0]

    def gate_passed(c: SectionCandidate) -> bool:
        return bool(c.summary.get("connection_gate_passed", True))

    def found_autonomous(c: SectionCandidate) -> bool:
        defect = _layer(c.summary, "coherence_defect")
        return bool(
            gate_passed(c)
            and int(c.summary.get("cycle_basis_size", 0)) >= max(1, int(gate_min_cycle_basis))
            and float(c.summary.get("coherence_defect_transition_closed_fraction", defect.get("closed_transition_fraction", 0.0))) >= 0.999
            and float(c.summary.get("coherence_defect_transition_determinism_accuracy", defect.get("transition_determinism_accuracy", 0.0))) >= 0.999
            and int(c.summary.get("coherence_defect_patterns", defect.get("n_patterns", 0))) > 1
        )

    def found_nontriv_recurrent(c: SectionCandidate) -> bool:
        return bool(gate_passed(c) and c.summary.get("coherence_defect_nontrivial_recurrent_quotient", False))


    if not gate_passed(final_best):
        verdict = "BLIND SECTION DYNAMICS BLOCKED DEGENERATE AUTONOMY: connection gate not preserved"
    elif found_nontriv_recurrent(final_best):
        verdict = "BLIND SECTION DYNAMICS FOUND NONTRIVIAL RECURRENT COHERENCE-DEFECT QUOTIENT"
    elif found_autonomous(final_best):
        verdict = "BLIND SECTION DYNAMICS FOUND AUTONOMOUS COHERENCE-DEFECT QUOTIENT"
    elif float(final_best.summary.get("coherent_state_fraction", 0.0)) > 0.0:
        verdict = "BLIND SECTION DYNAMICS PRESERVED CONNECTION AND IMPROVED COHERENCE, BUT NO AUTONOMOUS DEFECT QUOTIENT YET"
    else:
        verdict = "NO BLIND SECTION-DYNAMICS SIGNAL in this run"

    summary = dict(
        verdict=verdict,
        q=int(q),
        graph_mode=str(graph_mode),
        seed_mode=str(seed_mode),
        seed_winners=str(seed_winners),
        seed_winner_index=int(seed_winner_index),
        population=int(population),
        generations=int(generations),
        min_frame_ports=int(min_frame_ports),
        frame_coordinate_mode=str(frame_coordinate_mode),
        cycle_objective=str(cycle_objective),
        connection_weight=float(connection_weight),
        connection_gate=str(connection_gate),
        gate_min_live_edge_quotients=int(gate_min_live_edge_quotients),
        gate_min_true_frames=int(gate_min_true_frames),
        gate_min_true_transports=int(gate_min_true_transports),
        gate_min_cycle_basis=int(gate_min_cycle_basis),
        gate_min_global_chords=int(gate_min_global_chords),
        gate_min_global_holonomy=int(gate_min_global_holonomy),
        gate_fail_scale=float(gate_fail_scale),
        final_connection_gate_passed=gate_passed(final_best),
        all_time_connection_gate_passed=gate_passed(best_ever),
        final_connection_gate_progress=float(final_best.summary.get("connection_gate_progress", 0.0)),
        all_time_connection_gate_progress=float(best_ever.summary.get("connection_gate_progress", 0.0)),
        final_best_score=float(final_best.score),
        all_time_best_score=float(best_ever.score),
        final_connection_score=float(final_best.connection_score),
        all_time_connection_score=float(best_ever.connection_score),
        final_section_score=float(final_best.section_score),
        all_time_section_score=float(best_ever.section_score),
        # Backward-compatible names kept, but now gated and based on the
        # observer-level coherence-defect quotient rather than a collapsed
        # no-cycle residual layer.
        final_found_autonomous_coherent_residual_quotient=found_autonomous(final_best),
        all_time_found_autonomous_coherent_residual_quotient=found_autonomous(best_ever),
        final_found_nontrivial_recurrent_minimal_support_quotient=found_nontriv_recurrent(final_best),
        all_time_found_nontrivial_recurrent_minimal_support_quotient=found_nontriv_recurrent(best_ever),
        final_found_autonomous_coherence_defect_quotient=found_autonomous(final_best),
        all_time_found_autonomous_coherence_defect_quotient=found_autonomous(best_ever),
        final_found_nontrivial_recurrent_coherence_defect_quotient=found_nontriv_recurrent(final_best),
        all_time_found_nontrivial_recurrent_coherence_defect_quotient=found_nontriv_recurrent(best_ever),
        final_best_section_summary=final_best.summary,
        all_time_best_section_summary=best_ever.summary,
    )
    winners = [best_ever, final_best] + final_candidates[: max(1, min(10, len(final_candidates)))]
    return hist_df, winners, summary


def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def write_outputs(hist_df, winners: List[SectionCandidate], summary: Dict, out: str, winners_path: Optional[str]) -> None:
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
            payload.append(dict(
                system=c.system,
                score=c.score,
                connection_score=c.connection_score,
                section_score=c.section_score,
                summary=c.summary,
                graph_id=c.graph_id,
            ))
        with open(winners_path, "wb") as f:
            pickle.dump(payload, f)


def make_plot(hist_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(hist_df):
        ax.plot(hist_df["generation"], hist_df["best_score"], label="best total")
        ax.plot(hist_df["generation"], hist_df["mean_score"], label="mean total")
        ax.plot(hist_df["generation"], hist_df["best_section_score"], label="best section")
    ax.set_title(summary.get("verdict", "blind observer section dynamics"))
    ax.set_xlabel("generation")
    ax.set_ylabel("coherent residual consistency score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blind selection for coherent observer-section residual consistency")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--graph-mode", choices=[
        "frame_random_diamond", "componented", "er", "frame_flat_c2", "frame_twist_c2",
        "frame_theta_flat_c2", "frame_theta_twist_c2", "frame_random_theta",
        "frame_double_flat_c2", "frame_double_twist_c2", "frame_random_double_diamond"], default="frame_random_theta")
    p.add_argument("--vertices", type=int, default=11)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.16)
    p.add_argument("--inter-prob", type=float, default=0.35)
    p.add_argument("--extra-intra-prob", type=float, default=0.25)
    p.add_argument("--max-pred", type=int, default=4)
    p.add_argument("--seed-mode", choices=[
        "random", "flat_c2", "twist_c2", "randomized_flat_topology", "diamond_random_rules",
        "theta_flat_c2", "theta_twist_c2", "theta_random_rules",
        "double_flat_c2", "double_twist_c2", "double_random_rules"], default="theta_twist_c2")
    p.add_argument("--seed-winners", default="", help="optional winners pickle to seed from a selected connection")
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
    p.add_argument("--cycle-objective", choices=["best", "all"], default="all")
    p.add_argument("--max-local-support", type=int, default=2)
    p.add_argument("--max-gauge-flips", type=int, default=4096)
    p.add_argument("--connection-weight", type=float, default=0.40)
    p.add_argument("--connection-gate", choices=["auto", "off"], default="auto",
                   help="hard-gate D-stage scoring on preserving the selected observer-frame connection")
    p.add_argument("--gate-min-live-edge-quotients", type=int, default=0,
                   help="override auto gate target for live inter-observer edge quotients")
    p.add_argument("--gate-min-true-frames", type=int, default=0,
                   help="override auto gate target for true multi-port observer frames")
    p.add_argument("--gate-min-true-transports", type=int, default=0,
                   help="override auto gate target for true frame transports")
    p.add_argument("--gate-min-cycle-basis", type=int, default=0,
                   help="override auto gate target for cycle basis rank")
    p.add_argument("--gate-min-global-chords", type=int, default=0,
                   help="override auto gate target for global chord count")
    p.add_argument("--gate-min-global-holonomy", type=int, default=0,
                   help="override auto gate target for valid global holonomies")
    p.add_argument("--gate-fail-scale", type=float, default=0.49,
                   help="maximum score available to candidates that fail the connection gate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/blind_observer_section_dynamics.csv")
    p.add_argument("--winners", default="example_results/blind_observer_section_dynamics_winners.pkl")
    p.add_argument("--plot", default="example_results/fig_blind_observer_section_dynamics.png")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    max_pred = None if args.max_pred is None or int(args.max_pred) <= 0 else int(args.max_pred)
    max_mut = None if args.max_mutations is None or int(args.max_mutations) <= 0 else int(args.max_mutations)
    hist_df, winners, summary = evolve_observer_section_dynamics(
        q=int(args.q),
        graph_mode=str(args.graph_mode),
        vertices=int(args.vertices),
        components=int(args.components),
        edge_prob=float(args.edge_prob),
        inter_prob=float(args.inter_prob),
        extra_intra_prob=float(args.extra_intra_prob),
        max_pred=max_pred,
        seed_mode=str(args.seed_mode),
        seed_winners=str(args.seed_winners),
        seed_winner_index=int(args.seed_winner_index),
        population=int(args.population),
        generations=int(args.generations),
        elite=int(args.elite),
        mutation_rate=float(args.mutation_rate),
        min_mutations=int(args.min_mutations),
        max_mutations=max_mut,
        random_injection=float(args.random_injection),
        max_channel_inputs=int(args.max_channel_inputs),
        max_channel_backgrounds=int(args.max_channel_backgrounds),
        max_state_samples=int(args.max_state_samples),
        state_warmup=int(args.state_warmup),
        schedule_samples=int(args.schedule_samples),
        cycle_max_len=int(args.cycle_max_len),
        min_frame_ports=int(args.min_frame_ports),
        frame_coordinate_mode=str(args.frame_coordinate_mode),
        max_local_support=int(args.max_local_support),
        max_gauge_flips=int(args.max_gauge_flips),
        connection_weight=float(args.connection_weight),
        cycle_objective=str(args.cycle_objective),
        connection_gate=str(args.connection_gate),
        gate_min_live_edge_quotients=int(args.gate_min_live_edge_quotients),
        gate_min_true_frames=int(args.gate_min_true_frames),
        gate_min_true_transports=int(args.gate_min_true_transports),
        gate_min_cycle_basis=int(args.gate_min_cycle_basis),
        gate_min_global_chords=int(args.gate_min_global_chords),
        gate_min_global_holonomy=int(args.gate_min_global_holonomy),
        gate_fail_scale=float(args.gate_fail_scale),
        seed=int(args.seed),
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
