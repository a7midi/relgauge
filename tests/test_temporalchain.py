import numpy as np

from relgauge import temporalchain as T


def test_copy_temporal_chain_transports_full_label():
    rng = np.random.default_rng(123)
    tc = T.make_rule_temporal_chain(kB=2, kM=2, kA=2, q=4, w=1, rng=rng, rule_ensemble="copy")
    m = T.temporal_chain_measure(tc)
    assert m["s1_residual_qcoords"] == 1.0
    assert m["s2_residual_qcoords"] == 1.0
    assert m["transport_mi_over_Hs2"] > 0.99
    assert m["source_to_s2_mi_over_Hs2"] > 0.99
    assert m["live_transport_selected"] is True


def test_constant_temporal_chain_has_trivial_final_label():
    rng = np.random.default_rng(456)
    tc = T.make_rule_temporal_chain(kB=2, kM=2, kA=2, q=4, w=1, rng=rng, rule_ensemble="constant")
    m = T.temporal_chain_measure(tc)
    assert m["s1_residual_qcoords"] == 0.0
    assert m["s2_residual_qcoords"] == 0.0
    assert m["live_transport_selected"] is False


def test_temporal_chain_sweep_smoke():
    df = T.run_temporal_chain_sweep(q=3, n_instances=1, rule_ensembles=("copy", "random"), verbose=False)
    assert len(df) == 2
    summary = T.analyze_temporal_chain(df)
    assert "verdict" in summary
