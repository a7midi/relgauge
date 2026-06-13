import numpy as np

from relgauge import blindobserversectiondynamics as BSD
from relgauge import observerframebundle as OBF
from relgauge import observersectiondynamics as OSD


def test_section_consistency_score_does_not_reward_motion_or_nontriviality():
    base_bundle = dict(
        mean_edge_mi_norm=1.0,
        live_edge_quotient_fraction=1.0,
        true_live_frame_fraction=1.0,
        true_live_frame_transport_fraction=1.0,
        valid_holonomy_fraction=1.0,
        loop_automorphism_valid_fraction=1.0,
        global_chord_count=2,
        global_valid_holonomy=2,
    )
    s1 = dict(
        bundle_summary=base_bundle,
        coherent_state_fraction=0.25,
        coherent_transition_closed_fraction=1.0,
        coherent_transition_determinism_accuracy=1.0,
        minimal_support_transition_closed_fraction=1.0,
        cycle_syndrome_transition_closed_fraction=1.0,
        minimal_support_moving_transition_fraction=0.0,
        minimal_support_nontrivial_recurrent_quotient=False,
        global_nontrivial_holonomy=0,
        layer_summaries={
            "minimal_support": {"transition_determinism_accuracy": 1.0},
            "cycle_syndrome": {"transition_determinism_accuracy": 1.0},
        },
    )
    s2 = dict(s1)
    s2.update(
        minimal_support_moving_transition_fraction=1.0,
        minimal_support_nontrivial_recurrent_quotient=True,
        global_nontrivial_holonomy=2,
    )
    assert BSD.section_consistency_score(s1)[0] == BSD.section_consistency_score(s2)[0]


def test_seeded_section_dynamics_selection_smoke():
    hist_df, winners, summary = BSD.evolve_observer_section_dynamics(
        q=4,
        graph_mode="frame_random_theta",
        seed_mode="theta_twist_c2",
        population=3,
        generations=1,
        elite=1,
        mutation_rate=0.0,
        random_injection=0.0,
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        max_state_samples=128,
        schedule_samples=2,
        frame_coordinate_mode="local_charts",
        seed=5,
        verbose=False,
    )
    assert not hist_df.empty
    assert len(winners) >= 1
    assert "final_best_section_summary" in summary
    assert "best_coherent_state_fraction" in hist_df.columns
    assert "best_minimal_support_transition_closed_fraction" in hist_df.columns
    assert summary["all_time_connection_score"] > 0.5


def test_theta_control_section_score_has_coherent_states():
    sys = OBF.make_frame_theta_control(q=4, twists=(False, False, True))
    _p, _t, _q, osd_summary = OSD.run_section_dynamics(
        sys=sys,
        q=4,
        rng=np.random.default_rng(7),
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        max_state_samples=128,
        schedule_samples=2,
        frame_coordinate_mode="local_charts",
    )
    total, conn, sec = BSD.section_consistency_score(osd_summary)
    assert osd_summary["coherent_state_fraction"] > 0
    assert conn > 0.5
    assert total > 0
    assert sec > 0


def test_connection_gate_blocks_degenerate_autonomy_without_cycles():
    collapsed = dict(
        bundle_summary=dict(
            n_live_edge_quotients=3,
            n_true_live_frames=2,
            n_true_live_frame_transports=1,
            single_port_label_frame_fraction=0.4,
            full_boundary_violation=0,
        ),
        cycle_basis_size=0,
        global_chord_count=0,
        global_valid_holonomy=0,
        coherent_state_fraction=0.5,
        coherence_defect_transition_closed_fraction=1.0,
        coherence_defect_transition_determinism_accuracy=1.0,
        layer_summaries={
            "coherence_defect": {"closed_transition_fraction": 1.0, "transition_determinism_accuracy": 1.0},
            "minimal_support": {"transition_determinism_accuracy": 1.0},
            "cycle_syndrome": {"transition_determinism_accuracy": 1.0},
        },
    )
    passed, progress, _ = BSD.connection_gate_status(
        collapsed,
        gate_min_live_edge_quotients=6,
        gate_min_true_frames=5,
        gate_min_true_transports=6,
        gate_min_cycle_basis=2,
        gate_min_global_chords=2,
        gate_min_global_holonomy=2,
    )
    score, _conn, section = BSD.section_consistency_score(
        collapsed,
        require_connection_gate=True,
        gate_min_live_edge_quotients=6,
        gate_min_true_frames=5,
        gate_min_true_transports=6,
        gate_min_cycle_basis=2,
        gate_min_global_chords=2,
        gate_min_global_holonomy=2,
    )
    assert not passed
    assert progress < 1.0
    assert section > 0.75
    assert score < 0.49
