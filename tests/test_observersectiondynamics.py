from relgauge import observerframebundle as OBF
from relgauge import observersectiondynamics as OSD
import numpy as np


def test_single_nontrivial_diamond_reports_static_one_loop_frustration():
    sys = OBF.make_frame_diamond_control(q=4, twist=True)
    pattern_df, transition_df, quotient_df, summary = OSD.run_section_dynamics(
        sys=sys,
        q=4,
        rng=np.random.default_rng(0),
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        max_state_samples=256,
        schedule_samples=4,
        frame_coordinate_mode="local_charts",
    )
    assert "STATIC ONE-LOOP" in summary["verdict"]
    assert summary["global_nontrivial_holonomy"] == 1
    assert summary["n_live_transport_edges"] == 4
    assert not pattern_df.empty
    assert not transition_df.empty
    assert not quotient_df.empty


def test_theta_section_dynamics_is_multicycle_smoke():
    sys = OBF.make_frame_theta_control(q=4, twists=(False, False, True))
    pattern_df, transition_df, quotient_df, summary = OSD.run_section_dynamics(
        sys=sys,
        q=4,
        rng=np.random.default_rng(1),
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        max_state_samples=256,
        schedule_samples=4,
        frame_coordinate_mode="local_charts",
    )
    assert summary["global_chord_count"] >= 2
    assert summary["global_nontrivial_holonomy"] >= 1
    assert summary["n_live_transport_edges"] >= 6
    assert summary["n_residual_patterns"] >= 1
    assert not pattern_df.empty


def test_section_dynamics_v2_separates_coherent_cohomology_layers():
    sys = OBF.make_frame_theta_control(q=4, twists=(False, False, True))
    pattern_df, transition_df, quotient_df, summary = OSD.run_section_dynamics(
        sys=sys,
        q=4,
        rng=np.random.default_rng(2),
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        max_state_samples=256,
        schedule_samples=4,
        frame_coordinate_mode="local_charts",
    )
    assert summary["audit_version"] == "section_dynamics_v2_coherent_cohomology"
    assert summary["cycle_basis_size"] >= 2
    assert "connection_holonomy_bits" in summary
    assert summary["coherent_state_fraction"] > 0
    assert summary["cycle_syndrome_patterns"] >= 1
    assert summary["minimal_support_patterns"] >= 1
    assert set(pattern_df["representation"]) >= {"raw_port", "coherent_edge", "cycle_syndrome", "minimal_support"}
    assert set(transition_df["representation"]) >= {"raw_port"}
    assert set(quotient_df["representation"]) >= {"raw_port", "coherent_edge", "cycle_syndrome", "minimal_support"}


def test_minimal_support_representative_preserves_syndrome():
    edges = [(0, 1), (1, 2), (0, 3), (3, 2), (0, 4), (4, 2)]
    basis = OSD.cycle_basis_for_edges(edges)
    bits = (0, 0, 1, 0, 1, 0)
    min_bits, min_code, exhaustive = OSD.minimal_support_representative(bits, basis)
    assert exhaustive
    assert OSD.syndrome_bits(bits, basis) == OSD.syndrome_bits(min_bits, basis)
    assert sum(min_bits) <= sum(bits)
    assert min_code == OSD.residual_code(min_bits)
