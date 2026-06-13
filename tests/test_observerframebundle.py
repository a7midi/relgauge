from relgauge import observerframebundle as OBF


def test_frame_flat_c2_control_extracts_live_flat_connection():
    frame_df, edge_df, trans_df, cycle_df, summary = OBF.run_audit(
        q=4,
        graphs=1,
        graph_mode="frame_flat_c2",
        vertices=8,
        max_state_samples=1024,
        max_channel_inputs=64,
        max_channel_backgrounds=64,
        verbose=False,
    )
    assert summary["full_boundary_violations"] == 0
    assert summary["total_live_frames"] >= 4
    assert summary["total_single_port_label_frames"] == 0
    assert summary["total_live_frame_transports"] >= 4
    assert summary["total_valid_holonomy"] >= 1
    assert summary["total_global_valid_holonomy"] >= 1
    assert summary["group_counts"].get("trivial", 0) >= 1
    assert not frame_df.empty
    assert not edge_df.empty
    assert not trans_df.empty
    assert not cycle_df.empty


def test_frame_twist_c2_local_charts_detects_nonflat_c2():
    frame_df, edge_df, trans_df, cycle_df, summary = OBF.run_audit(
        q=4,
        graphs=1,
        graph_mode="frame_twist_c2",
        vertices=8,
        max_state_samples=1024,
        max_channel_inputs=64,
        max_channel_backgrounds=64,
        frame_coordinate_mode="local_charts",
        verbose=False,
    )
    assert summary["full_boundary_violations"] == 0
    assert summary["total_true_live_frames"] >= 4
    assert summary["total_true_live_frame_transports"] >= 4
    assert summary["total_valid_holonomy"] >= 1
    assert summary["total_nontrivial_holonomy"] >= 1
    assert summary["total_global_nontrivial_holonomy"] >= 1
    assert summary["group_counts"].get("C2", 0) >= 1
    assert summary["holonomy_type_counts"].get("cycle_2", 0) >= 1
    assert float(cycle_df["path_agreement"].max()) == 0.0


def test_frame_twist_c2_common_factor_still_flattens():
    _frame_df, _edge_df, _trans_df, _cycle_df, summary = OBF.run_audit(
        q=4,
        graphs=1,
        graph_mode="frame_twist_c2",
        vertices=8,
        max_state_samples=1024,
        max_channel_inputs=64,
        max_channel_backgrounds=64,
        frame_coordinate_mode="common_factor",
        verbose=False,
    )
    assert summary["total_valid_holonomy"] >= 1
    assert summary["total_nontrivial_holonomy"] == 0
    assert summary["group_counts"].get("trivial", 0) >= 1


def test_frame_random_diamond_is_valid_negative_smoke():
    frame_df, edge_df, trans_df, cycle_df, summary = OBF.run_audit(
        q=4,
        graphs=1,
        graph_mode="frame_random_diamond",
        vertices=8,
        max_state_samples=256,
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        seed=7,
        verbose=False,
    )
    assert summary["full_boundary_violations"] == 0
    assert "verdict" in summary
    assert set(["frame_classes", "frame_live", "port_label_live", "true_frame_live"]).issubset(frame_df.columns)
    assert set(["edge_mi_norm", "live_edge_quotient"]).issubset(edge_df.columns)
    assert set(["transport_accuracy", "live_frame_transport", "port_label_transport", "true_frame_transport"]).issubset(trans_df.columns)
    assert set(["usable_diamond", "path_agreement", "best_branch_completion", "two_branch_completion"]).issubset(cycle_df.columns)
    assert "max_branch_completion" in summary
    assert "global_valid_holonomy" in summary


def test_nonflat_loop_counts_as_automorphism_valid_not_flat():
    def tr(u, v, mapping):
        return OBF.FrameTransport(
            comp_edge=(u, v),
            source_frame=u,
            target_frame=v,
            n_source_classes=2,
            n_target_classes=2,
            n_source_ports=2,
            n_target_ports=2,
            mapping=mapping,
            accuracy=1.0,
            coverage_fraction=1.0,
            bijective=True,
            source_entropy_bits=1.0,
            live=True,
            true_live=True,
        )

    transports = {
        (0, 1): tr(0, 1, {0: 0, 1: 1}),
        (1, 3): tr(1, 3, {0: 0, 1: 1}),
        (0, 2): tr(0, 2, {0: 0, 1: 1}),
        (2, 3): tr(2, 3, {0: 1, 1: 0}),
    }
    row = OBF.analyze_frame_cycle((0, 1, 3, 2), transports, set(transports))
    assert row["usable_diamond"] == 1
    assert row["path_agreement"] == 0.0
    assert row["valid_connection"] == 0
    assert row["flat_connection"] == 0
    assert row["loop_automorphism_valid"] == 1
    assert row["valid_holonomy"] == 1
    assert row["nontrivial_holonomy"] == 1
    assert row["generated_group"] == "C2"


def test_theta_twist_control_has_multiple_cycles_and_c2_holonomy():
    _frame_df, _edge_df, _trans_df, cycle_df, summary = OBF.run_audit(
        q=4,
        graphs=1,
        graph_mode="frame_theta_twist_c2",
        max_state_samples=1024,
        max_channel_inputs=64,
        max_channel_backgrounds=64,
        frame_coordinate_mode="local_charts",
        verbose=False,
    )
    assert summary["total_true_live_frames"] >= 5
    assert summary["total_true_live_frame_transports"] >= 6
    assert summary["total_valid_holonomy"] >= 3
    assert summary["total_nontrivial_holonomy"] >= 2
    assert summary["total_global_nontrivial_holonomy"] >= 1
    assert summary["group_counts"].get("C2", 0) >= 1
    assert len(cycle_df) >= 3


def test_double_diamond_twist_control_has_two_loop_arena():
    _frame_df, _edge_df, _trans_df, cycle_df, summary = OBF.run_audit(
        q=4,
        graphs=1,
        graph_mode="frame_double_twist_c2",
        max_state_samples=1024,
        max_channel_inputs=64,
        max_channel_backgrounds=64,
        frame_coordinate_mode="local_charts",
        verbose=False,
    )
    assert summary["total_true_live_frames"] >= 7
    assert summary["total_true_live_frame_transports"] >= 8
    assert summary["total_valid_holonomy"] >= 2
    assert summary["total_global_valid_holonomy"] >= 2
    assert summary["total_global_nontrivial_holonomy"] >= 1
    assert len(cycle_df) >= 2
