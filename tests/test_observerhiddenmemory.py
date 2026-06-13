import numpy as np

from relgauge import observerframebundle as OBF
from relgauge import observerhiddenmemory as OHM


def test_hidden_memory_audit_theta_smoke():
    sys = OBF.make_frame_theta_control(q=4, twists=(False, False, True))
    pattern_df, transition_df, quotient_df, observer_df, summary = OHM.run_hidden_memory_audit(
        sys=sys,
        q=4,
        rng=np.random.default_rng(0),
        graph_mode="frame_random_theta",
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        max_state_samples=128,
        schedule_samples=2,
        frame_coordinate_mode="local_charts",
        connection_gate="auto",
    )
    assert summary["audit_version"] == "observer_hidden_memory_v3_visible_quotient_period_audit"
    assert summary["connection_gate_passed"]
    assert summary["distinct_visible_states"] >= 1
    assert summary["distinct_joint_states"] >= summary["distinct_visible_states"]
    assert "hidden_conditioned_visible_determinism_accuracy" in summary
    assert not pattern_df.empty
    assert not transition_df.empty
    assert not quotient_df.empty
    assert not observer_df.empty


def test_color_preserving_quotient_preserves_visible_colors():
    # 0(color A) has schedule-ambiguous successors 1 and 2 of same color B;
    # the quotient should merge 1 and 2 without merging visible colors A/B.
    nodes = {0, 1, 2}
    edges = {0: {1, 2}, 1: {1}, 2: {2}}
    colors = {0: 0, 1: 1, 2: 1}
    counts = {0: 10, 1: 5, 2: 5}
    qmap, summary, rows = OHM.color_preserving_deterministic_quotient(nodes, edges, colors, counts)
    assert qmap[1] == qmap[2]
    assert qmap[0] != qmap[1]
    assert summary["source_block_deterministic_fraction"] == 1.0
    assert summary["visible_conflict_source_fraction"] == 0.0
    assert len(rows) == 2


def test_deterministic_schedule_quotient_allows_visible_merging():
    # Visible state 0 has schedule-ambiguous successors 1 and 2. The visible
    # quotient may merge 1 and 2, unlike the color-preserving memory quotient.
    nodes = {0, 1, 2}
    edges = {0: {1, 2}, 1: {1}, 2: {2}}
    counts = {0: 10, 1: 5, 2: 5}
    qmap, summary, rows = OHM.deterministic_schedule_quotient(nodes, edges, counts)
    assert qmap[1] == qmap[2]
    assert qmap[0] != qmap[1]
    assert summary["source_block_deterministic_fraction"] == 1.0
    assert summary["quotient_classes"] == 2
    assert len(rows) == 2



def test_recurrent_period_summary_detects_two_cycle():
    counts = {0: 5, 1: 5}
    edges = {0: {1}, 1: {0}}
    summary = OHM.quotient_transition_summary(counts, edges)
    assert summary["source_block_deterministic_fraction"] == 1.0
    assert summary["recurrent_component_count"] == 1
    assert summary["recurrent_state_count"] == 2
    assert summary["recurrent_cycle_lengths"] == [2]
    assert summary["max_recurrent_period"] == 2
    assert summary["fixed_point_count"] == 0
    assert summary["nontrivial_periodic_recurrence"] is True


def test_deterministic_schedule_quotient_reports_fixed_point_period():
    nodes = {0, 1, 2}
    edges = {0: {1, 2}, 1: {1}, 2: {2}}
    counts = {0: 10, 1: 5, 2: 5}
    _qmap, summary, _rows = OHM.deterministic_schedule_quotient(nodes, edges, counts)
    assert summary["quotient_classes"] == 2
    assert summary["source_block_deterministic_fraction"] == 1.0
    assert summary["max_recurrent_period"] == 1
    assert summary["fixed_point_count"] == 1
    assert summary["nontrivial_periodic_recurrence"] is False
