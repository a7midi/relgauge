import pickle

from relgauge import observerframebundle as OBF
from relgauge import observerconnectioncertifier as OCC
from relgauge import observerconnectionsuccessaudit as OCSA


def test_certifier_accepts_twist_control_q2(tmp_path):
    sys = OBF.make_frame_diamond_control(q=2, twist=True)
    winners = tmp_path / "winners.pkl"
    with winners.open("wb") as f:
        pickle.dump([{"system": sys, "score": 0.0, "summary": {}, "graph_id": 0}], f)

    frame_df, edge_df, trans_df, cycle_df, check_df, result = OCC.certify_winners(
        str(winners),
        winner_index=0,
        min_frame_ports=2,
        frame_coordinate_mode="local_charts",
        require_nontrivial=True,
        require_c2=True,
        require_nonflat=True,
        max_exhaustive_states=10000,
    )
    assert result["aggregate"]["n_certified_passed"] == 1
    assert result["summary_rows"][0]["global_generated_group"] == "C2"
    assert result["summary_rows"][0]["global_nontrivial_holonomy"] > 0
    assert int(check_df[check_df["check"] == "nontrivial_holonomy"]["passed"].iloc[0]) == 1


def test_success_audit_smoke_twist_seed_q2():
    seed_df, hist_df, winners, summary = OCSA.run_success_audit(
        q=2,
        runs=2,
        seed_start=10,
        graph_mode="frame_random_diamond",
        seed_mode="twist_c2",
        population=4,
        generations=1,
        elite=1,
        mutation_rate=0.0,
        random_injection=0.0,
        max_channel_inputs=64,
        max_channel_backgrounds=64,
        max_state_samples=256,
        frame_coordinate_mode="local_charts",
        verbose=False,
    )
    assert len(seed_df) == 2
    assert len(hist_df) > 0
    assert len(winners) == 4  # all-time and final per run
    assert summary["aggregate"]["all_time_connection_fraction"] == 1.0
    assert summary["aggregate"]["all_time_nontrivial_fraction"] == 1.0
