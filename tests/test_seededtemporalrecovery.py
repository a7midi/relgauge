import numpy as np

from relgauge import seededtemporalrecovery as S


def test_seeded_candidate_evaluates():
    rng = np.random.default_rng(0)
    tc = S.make_seeded_candidate(2, 2, 2, 3, 1, rng, "copy", 0.0)
    m = S.evaluate_recovery_candidate(tc, initial_mode="source_all", n_random_initial=64)
    assert m["s1_residual_qcoords"] > 0
    assert m["s2_residual_qcoords"] > 0
    assert m["transport_mi_over_Hs2"] >= 0
    assert m["source_to_s2_mi_over_Hs2"] >= 0


def test_seeded_recovery_smoke():
    df, saved = S.run_seeded_temporal_recovery(
        ks_B=(2,), ks_M=(2,), ks_A=(2,), ws=(1,), q=3,
        seed_ensembles=("copy",), seed_mutation_rates=(0.0,),
        runs=1, population=3, generations=1,
        initial_mode="source_all", n_random_initial=32,
        verbose=False,
    )
    assert len(df) > 0
    assert len(saved) > 0
    res = S.analyze_seeded_recovery(df)
    assert "verdict" in res
