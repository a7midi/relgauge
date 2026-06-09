import numpy as np

from relgauge import redundancytemporal as R


def test_repeat_candidate_has_nonzero_transport_signal():
    rng = np.random.default_rng(123)
    tc = R.make_redundancy_candidate(2, 2, 4, 4, 2, rng, "repeat_copy", 0.0)
    m = R.evaluate_candidate(tc, initial_mode="source_all", n_random_initial=128, rng=rng)
    assert m["s1_residual_qcoords"] > 0
    assert m["s2_residual_qcoords"] > 0
    assert m["transport_mi_over_Hs2"] > 0.9
    assert m["source_to_s2_mi_over_Hs2"] > 0.9


def test_smoke_redundancy_sweep_runs():
    df, saved = R.run_redundancy_temporal_recovery(
        ks_B=(2,), ks_M=(2,), ks_A=(4,), ws=(2,), q=3,
        seed_ensembles=("repeat_copy", "capacity_copy", "random"),
        seed_mutation_rates=(0.05,),
        runs=1, population=3, generations=1,
        n_random_initial=256, verbose=False,
    )
    assert len(df) > 0
    summary = R.analyze_redundancy_recovery(df)
    assert "verdict" in summary
    assert isinstance(saved, list)
