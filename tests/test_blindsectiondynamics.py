import numpy as np

from relgauge import observerframebundle as OBF
from relgauge import observersectiondynamics as OSD
from relgauge import blindsectiondynamics as BSD


def test_section_consistency_score_is_defined_for_theta_control():
    sys = OBF.make_frame_theta_control(q=4, twists=(False, False, True))
    _pdf, _tdf, _qdf, summary = OSD.run_section_dynamics(
        sys=sys,
        q=4,
        rng=np.random.default_rng(0),
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        max_state_samples=128,
        schedule_samples=3,
        frame_coordinate_mode="local_charts",
    )
    score = BSD.score_from_section_summary(summary, coherence_target=0.01)
    assert 0.0 <= score <= 1.0
    assert BSD.connection_consistency(summary) > 0.8


def test_blind_section_dynamics_seeded_smoke():
    hist_df, winners, summary = BSD.evolve_section_dynamics(
        q=4,
        graph_mode="frame_random_theta",
        seed_mode="theta_twist_c2",
        population=3,
        generations=1,
        elite=1,
        mutation_rate=0.02,
        max_channel_inputs=32,
        max_channel_backgrounds=32,
        max_state_samples=128,
        schedule_samples=3,
        frame_coordinate_mode="local_charts",
        coherence_target=0.01,
        seed=1,
        verbose=False,
    )
    assert not hist_df.empty
    assert len(winners) >= 1
    assert 0.0 <= summary["final_best_score"] <= 1.0
    assert "final_best_summary" in summary
