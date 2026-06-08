"""Tests for finite observer phase-channel experiments."""
import numpy as np

from relgauge import finiteobserver as F
from relgauge import phasechannel as P


def test_schedule_groups_by_cut_cover_all_schedules():
    rng = np.random.default_rng(0)
    co = F.make_coupled_observer(kB=3, kA=2, q=3, w=1, rng=rng)
    schedules, groups = P.schedule_groups_by_cut(co)
    covered = sorted(i for g in groups.values() for i in g.tolist())
    assert covered == list(range(len(schedules)))
    assert set(groups) == {0, 1, 2}
    # For each first B vertex: (kB-1)! internal B orders times kA! A orders.
    assert all(len(groups[v]) == 2 * 2 for v in groups)


def test_phase_channel_measure_obeys_bounds():
    rng = np.random.default_rng(1)
    co = F.make_coupled_observer(kB=3, kA=2, q=3, w=1, rng=rng)
    m = P.phase_channel_measure(co, horizon=2, mode="cut", protocol="first", do_bulk=True)
    eps = 1e-7
    assert m["phase_capacity_bits"] <= m["phase_entropy_bits"] + eps
    assert m["phase_capacity_bits"] <= m["memory_bits"] + eps
    assert m["phase_capacity_bits"] <= m["physical_bound_bits"] + eps
    assert m["phase_zero_error_bits"] <= m["phase_entropy_bits"] + eps
    assert m["phase_zero_error_bits"] <= m["memory_bits"] + eps
    assert m["n_phase_labels"] == 3


def test_erased_value_phase_channel_smoke():
    rng = np.random.default_rng(2)
    co = F.make_coupled_observer(kB=3, kA=2, q=3, w=1, rng=rng)
    m = P.phase_channel_measure(co, horizon=1, mode="erased_value", protocol="first", phase_vertex=0)
    assert 1 <= m["n_phase_labels"] <= 3
    assert m["phase_capacity_bits"] >= 0.0
    assert m["phase_capacity_bits"] <= m["phase_entropy_bits"] + 1e-7


def test_phase_channel_sweep_smoke_and_analysis():
    df = P.run_phase_channel_sweep(
        ks_B=(2, 3),
        ks_A=(2,),
        ws=(1,),
        horizons=(1,),
        q=3,
        n_instances=2,
        phase_modes=("cut", "erased_value"),
        protocols=("first",),
        base_seed=3,
        max_joint_states=3 ** 5,
        do_bulk=True,
        verbose=False,
    )
    assert len(df) == 8
    res = P.analyze_phase_channel(df)
    assert res["phase_bound_violations"] == 0
    assert res["phase_memory_bound_violations"] == 0
    assert res["phase_entropy_bound_violations"] == 0
    assert res["zero_error_phase_entropy_bound_violations"] == 0
