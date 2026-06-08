import math

import numpy as np

from relgauge import ruleselection as R


def test_copy_rule_ensemble_has_full_shared_interface():
    rng = np.random.default_rng(123)
    ds = R.make_rule_selection_diamond(2, 2, 2, 3, 1, rng, rule_ensemble="copy")
    m = R.rule_selection_measure(ds, eq_mode="joint", initial_pool="all")
    assert m["eq_shared_classes_mean"] == 3
    assert math.isclose(m["eq_residual_qcoords"], 1.0)
    assert m["selected"] is True


def test_constant_rule_ensemble_has_trivial_interface():
    rng = np.random.default_rng(123)
    ds = R.make_rule_selection_diamond(2, 2, 2, 3, 1, rng, rule_ensemble="constant")
    m = R.rule_selection_measure(ds, eq_mode="joint", initial_pool="all")
    assert m["eq_shared_classes_mean"] == 1
    assert math.isclose(m["eq_residual_qcoords"], 0.0)
    assert m["selected"] is False


def test_rule_selection_sweep_separates_random_from_selected_controls():
    df = R.run_rule_selection_sweep(
        ks_B=(2,),
        ks_M=(2,),
        ks_A=(2,),
        ws=(1,),
        q=3,
        n_instances=1,
        rule_ensembles=("random", "copy", "constant"),
        mutation_rates=(0.0,),
        eq_modes=("joint",),
        verbose=False,
    )
    res = R.analyze_rule_selection(df)
    assert res["equalizer_bound_violations"] == 0
    copy_rows = df[df.rule_ensemble == "copy"]
    const_rows = df[df.rule_ensemble == "constant"]
    assert float(copy_rows.eq_residual_qcoords.iloc[0]) > 0.99
    assert float(const_rows.eq_residual_qcoords.iloc[0]) == 0.0
