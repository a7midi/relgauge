import numpy as np

from relgauge import approxequalizer as AE
from relgauge import ruleselection as R
from relgauge import scdcdiamond as S


def test_noisy_diagonal_exact_collapses_but_approx_recovers():
    q = 4
    M = np.ones((q, q), dtype=np.int64)
    np.fill_diagonal(M, 100)
    s0 = AE.approximate_shared_equalizer_from_matrix(M, q=q, width=1, threshold=0.0)
    s1 = AE.approximate_shared_equalizer_from_matrix(M, q=q, width=1, threshold=0.2)
    assert s0.exact_shared_classes == 1
    assert s1.approx_classes == q
    assert s1.approx_residual_qcoords == 1.0
    assert s1.approx_info_qcoords > 0.7


def test_random_uniform_stays_trivial_under_conditional_threshold():
    q = 4
    M = np.ones((q, q), dtype=np.int64)
    s = AE.approximate_shared_equalizer_from_matrix(M, q=q, width=1, threshold=0.2)
    assert s.approx_classes == 1
    assert s.approx_residual_qcoords == 0.0
    assert abs(s.approx_info_qcoords) < 1e-12


def test_copy_rule_has_full_approx_equalizer():
    rng = np.random.default_rng(3)
    ds = S.make_scdc_diamond(2, 2, 2, 2, 4, 1, rng)
    R.apply_rule_ensemble(ds, "copy", rng)
    rows = AE.approx_equalizer_measure(ds, thresholds=(0.2,), initial_pool="all")
    assert len(rows) == 1
    row = rows[0]
    assert row["aeq_approx_classes"] == 4
    assert abs(row["aeq_approx_residual_qcoords"] - 1.0) < 1e-12
    assert row["aeq_approx_info_qcoords"] > 0.95


def test_sweep_smoke():
    df = AE.run_approx_equalizer_sweep(
        q=3,
        ks_B=(2,),
        ks_M=(2,),
        ks_A=(2,),
        ws=(1,),
        rule_ensembles=("random", "copy"),
        mutation_rates=(0.0,),
        thresholds=(0.2,),
        n_instances=2,
        verbose=False,
    )
    assert not df.empty
    assert set(df.rule_ensemble) == {"random", "copy"}
    res = AE.analyze_approx_equalizer(df)
    assert "verdict" in res
