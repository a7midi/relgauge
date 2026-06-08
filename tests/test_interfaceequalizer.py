import math

import numpy as np

from relgauge import interfaceequalizer as IE
from relgauge import scdcdiamond as S


def test_maximal_shared_equalizer_diagonal_preserves_one_q_coordinate():
    q = 4
    left = np.arange(q, dtype=np.int64)
    right = np.arange(q, dtype=np.int64)
    eq = IE.maximal_shared_equalizer(left, right, q=q, width=1)
    assert eq.shared_classes == q
    assert abs(eq.residual_qcoords - 1.0) < 1e-12
    assert abs(eq.cost_qcoords - 1.0) < 1e-12


def test_maximal_shared_equalizer_complete_bipartite_collapses_interface():
    q = 4
    left = np.repeat(np.arange(q, dtype=np.int64), q)
    right = np.tile(np.arange(q, dtype=np.int64), q)
    eq = IE.maximal_shared_equalizer(left, right, q=q, width=1)
    assert eq.shared_classes == 1
    assert abs(eq.residual_qcoords - 0.0) < 1e-12
    assert abs(eq.cost_qcoords - 2.0) < 1e-12


def test_interface_equalizer_measure_bounds_on_small_diamond():
    rng = np.random.default_rng(123)
    ds = S.make_scdc_diamond(kB=2, kC=2, kD=2, kA=2, q=3, w=1, rng=rng)
    m = IE.interface_equalizer_measure(ds, eq_mode="pairwise", initial_pool="all")
    assert 0.0 <= m["eq_residual_qcoords"] <= m["w"] + 1e-12
    assert 0.0 <= m["eq_cost_qcoords"] <= 2 * m["w"] + 1e-12
    assert m["product_cost_A_qcoords"] >= 0.0
    assert "overcollapse_avoided_qcoords" in m


def test_run_interface_equalizer_sweep_and_analyze():
    df = IE.run_interface_equalizer_sweep(
        ks_B=(2,),
        ks_M=(2,),
        ks_A=(2,),
        ws=(1,),
        q=3,
        n_instances=2,
        eq_modes=("pairwise", "joint"),
        initial_pool="all",
        max_joint_states=3**8,
        verbose=False,
    )
    assert len(df) == 4
    res = IE.analyze_interface_equalizer(df)
    assert res["equalizer_bound_violations"] == 0
    assert "verdict" in res
