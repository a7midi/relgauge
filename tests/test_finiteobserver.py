"""Tests for the finite SCC observer channel correction."""
import numpy as np

from relgauge import finiteobserver as F


def test_coupled_observer_has_two_distinct_feedback_sccs():
    rng = np.random.default_rng(0)
    co = F.make_coupled_observer(kB=3, kA=2, q=3, w=1, rng=rng)
    assert F.nontrivial_scc_count(co.joint) == 2
    # Interface is feed-forward only: no A vertex can be a predecessor of B.
    for v in range(co.kB):
        assert all(p < co.kB for p in co.joint.preds[v])


def test_finite_observer_capacity_obeys_memory_bound():
    rng = np.random.default_rng(1)
    co = F.make_coupled_observer(kB=3, kA=2, q=3, w=1, rng=rng)
    m = F.finite_observer_measure(co, horizon=2, observer_init="zero", do_shannon=True)
    eps = 1e-7
    assert m["shannon_capacity_bits"] <= m["memory_bits"] + eps
    assert m["zero_error_bits"] <= m["memory_bits"] + eps
    assert m["shannon_capacity_bits"] <= m["logRecB_bits"] + eps
    assert m["zero_error_bits"] <= m["logRecB_bits"] + eps


def test_finite_observer_sweep_smoke_and_analysis():
    df = F.run_finite_observer_sweep(
        ks_B=(2, 3),
        ks_A=(2,),
        ws=(1,),
        horizons=(1,),
        q=3,
        n_instances=2,
        base_seed=2,
        max_joint_states=3 ** 5,
        do_shannon=True,
        verbose=False,
    )
    assert len(df) == 4
    res = F.analyze_finite_observer(df)
    assert res["memory_bound_violations"] == 0
    assert res["zero_error_memory_bound_violations"] == 0
    assert res["physical_bound_violations"] == 0
