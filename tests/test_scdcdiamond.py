import numpy as np

from relgauge import scdcdiamond as S


def test_scdc_diamond_builder_and_bounds():
    rng = np.random.default_rng(0)
    ds = S.make_scdc_diamond(2, 2, 2, 2, 3, 1, rng)
    assert ds.k_total == 8
    assert ds.panel_pairs == ((6, 7),)
    m = S.scdc_diamond_quotient_cost(ds, constraint_mode="sink_equal", initial_pool="all")
    assert 0.0 <= m["quotient_cost_qcoords"] <= ds.k_total
    assert m["cost_A_qcoords"] >= 0.0
    assert m["collapsed_vertices"] >= 0


def test_branch_order_confluence_cost_zero_for_feedforward_diamond():
    rng = np.random.default_rng(1)
    ds = S.make_scdc_diamond(2, 2, 2, 2, 3, 1, rng)
    # C and D have no edges between them and A updates after both, so the paired
    # CD/DC branch-order maps are exactly equal before quotienting.
    m = S.scdc_diamond_quotient_cost(ds, constraint_mode="branch_order", initial_pool="all")
    assert m["branch_order_seed_merges"] == 0
    assert abs(m["quotient_cost_qcoords"]) < 1e-12


def test_scdc_diamond_sweep_and_analysis_smoke():
    df = S.run_scdc_diamond_sweep(
        ks_B=(2,),
        ks_M=(2,),
        ks_A=(2,),
        ws=(1,),
        q=3,
        n_instances=2,
        constraint_modes=("branch_order", "sink_equal"),
        initial_pool="all",
        max_joint_states=3 ** 8,
        verbose=False,
    )
    assert len(df) == 4
    res = S.analyze_scdc_diamond(df)
    assert "verdict" in res
    assert res["cost_bound_violations"] == 0
    assert "sink_equal" in res["by_mode"]
