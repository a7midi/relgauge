from relgauge import blindobserverconnection as B


def test_seeded_observer_connection_recovery_smoke():
    hist_df, winners, summary = B.evolve_observer_connection(
        q=4,
        graph_mode="frame_random_diamond",
        seed_mode="flat_c2",
        population=4,
        generations=1,
        elite=1,
        max_state_samples=256,
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        seed=11,
        verbose=False,
    )
    assert not hist_df.empty
    assert len(winners) >= 1
    assert summary["all_time_found_connection"] is True
    assert summary["all_time_found_valid_holonomy"] is True
    assert summary["all_time_found_global_holonomy"] is True
    assert summary["all_time_two_branch_completion"] == 1.0
    assert summary["all_time_best_score"] > 0.9


def test_seeded_twist_connection_recovery_reports_nontrivial_holonomy():
    hist_df, winners, summary = B.evolve_observer_connection(
        q=4,
        graph_mode="frame_random_diamond",
        seed_mode="twist_c2",
        population=4,
        generations=1,
        elite=1,
        max_state_samples=256,
        max_channel_inputs=64,
        max_channel_backgrounds=64,
        frame_coordinate_mode="local_charts",
        seed=13,
        verbose=False,
    )
    assert not hist_df.empty
    assert len(winners) >= 1
    assert summary["all_time_found_automorphism_valid_connection"] is True
    assert summary["all_time_found_nontrivial_holonomy"] is True
    assert summary["all_time_found_global_nontrivial_holonomy"] is True
    assert summary["all_time_path_agreement"] == 0.0


def test_blind_random_observer_connection_smoke_runs_without_group_target():
    hist_df, winners, summary = B.evolve_observer_connection(
        q=4,
        graph_mode="frame_random_diamond",
        seed_mode="random",
        population=3,
        generations=1,
        elite=1,
        max_state_samples=128,
        max_channel_inputs=16,
        max_channel_backgrounds=16,
        seed=12,
        verbose=False,
    )
    assert not hist_df.empty
    assert len(winners) >= 1
    assert "final_best_summary" in summary
    assert "best_mean_edge_mi_norm" in hist_df.columns
    assert "best_true_live_frame_fraction" in hist_df.columns
    assert "best_max_branch_completion" in hist_df.columns
    assert "best_global_valid_holonomy" in hist_df.columns


def test_v3_score_rewards_loop_automorphism_not_path_flatness():
    base = dict(
        mean_edge_mi_norm=1.0,
        live_edge_quotient_fraction=1.0,
        true_live_frame_fraction=1.0,
        true_live_frame_transport_fraction=1.0,
        max_branch_completion=1.0,
        max_two_branch_completion=1.0,
        max_complete_branch=1.0,
        n_usable_diamonds=1,
    )
    auto = dict(base, max_loop_automorphism_validity=1.0, max_path_agreement=0.0)
    flat_only = dict(base, max_loop_automorphism_validity=0.0, max_path_agreement=1.0)
    assert abs(B.score_from_summary(auto) - 1.0) < 1e-12
    assert abs(B.score_from_summary(flat_only) - 0.9) < 1e-12


def test_seeded_theta_connection_all_cycle_objective_smoke():
    hist_df, winners, summary = B.evolve_observer_connection(
        q=4,
        graph_mode="frame_random_theta",
        seed_mode="theta_twist_c2",
        population=3,
        generations=1,
        elite=1,
        max_state_samples=256,
        max_channel_inputs=64,
        max_channel_backgrounds=64,
        frame_coordinate_mode="local_charts",
        cycle_objective="all",
        seed=21,
        verbose=False,
    )
    assert not hist_df.empty
    assert len(winners) >= 1
    assert summary["all_time_found_automorphism_valid_connection"] is True
    assert summary["all_time_found_global_nontrivial_holonomy"] is True
    assert summary["all_time_best_summary"]["global_chord_count"] >= 2
