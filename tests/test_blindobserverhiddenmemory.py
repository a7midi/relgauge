from relgauge import blindobserverhiddenmemory as BHM


def test_hidden_memory_score_does_not_reward_recurrence_or_motion():
    base = dict(
        bundle_summary=dict(
            mean_edge_mi_norm=1.0,
            live_edge_quotient_fraction=1.0,
            true_live_frame_fraction=1.0,
            true_live_frame_transport_fraction=1.0,
            valid_holonomy_fraction=1.0,
            loop_automorphism_valid_fraction=1.0,
            global_chord_count=2,
            global_valid_holonomy=2,
        ),
        connection_gate_passed=True,
        connection_gate_progress=1.0,
        visible_only_transition_determinism_accuracy=0.4,
        hidden_conditioned_visible_determinism_accuracy=0.9,
        hidden_memory_ambiguity_reduction=0.8,
        memory_quotient_source_block_deterministic_fraction=0.9,
        memory_quotient_visible_conflict_source_fraction=0.0,
        memory_extra_fraction=0.5,
        memory_compression_ratio=0.5,
        memory_quotient_nontrivial_recurrent=False,
        memory_quotient_visible_moving_edge_fraction=0.0,
    )
    moved = dict(base)
    moved.update(memory_quotient_nontrivial_recurrent=True, memory_quotient_visible_moving_edge_fraction=1.0)
    assert BHM.memory_consistency_score(base)[0] == BHM.memory_consistency_score(moved)[0]


def test_blind_hidden_memory_smoke():
    hist_df, winners, summary = BHM.evolve_observer_hidden_memory(
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
        seed=3,
        verbose=False,
    )
    assert not hist_df.empty
    assert len(winners) == 2
    assert "final_best_hidden_summary" in summary
    assert "best_hidden_conditioned_visible_determinism" in hist_df.columns
    assert summary["all_time_connection_score"] > 0.5
