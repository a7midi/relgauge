import numpy as np

from relgauge import temporalchain as T
from relgauge import blindtemporalchain as B


def test_target_vertices_nonempty():
    rng = np.random.default_rng(1)
    tc = T.make_temporal_chain(2, 2, 2, 4, 1, rng)
    assert len(B.target_vertices(tc, "interface")) > 0
    assert len(B.target_vertices(tc, "all")) == tc.k_total


def test_evaluate_copy_chain_selected():
    rng = np.random.default_rng(2)
    tc = T.make_rule_temporal_chain(2, 2, 2, 4, 1, rng, rule_ensemble="copy")
    m = B.evaluate_candidate(tc, initial_mode="source_all", n_random_initial=16, rng=rng, stage="transport")
    assert m["min_residual_qcoords"] >= 1.0 - 1e-9
    assert m["transport_mi_over_Hs2"] >= 0.99
    assert m["source_to_s2_mi_over_Hs2"] >= 0.99
    assert m["selected_blind_live_transport"] is True


def test_blind_temporal_smoke_runs():
    df, winners = B.run_blind_temporal_chain(
        ks_B=(2,), ks_M=(2,), ks_A=(2,), ws=(1,), q=3,
        null_samples=2, runs=1, population=3, generations=1,
        fitness_mode="transport", initial_mode="source_all", n_random_initial=16,
        verbose=False,
    )
    assert not df.empty
    summary = B.analyze_blind_temporal_chain(df)
    assert "verdict" in summary
