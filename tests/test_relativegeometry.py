import numpy as np

from relgauge import relativegeometry as R


def test_full_closed_system_has_empty_boundary_and_proper_subsets_do_not():
    rng = np.random.default_rng(1)
    sys = R.make_closed_system(k=4, q=2, rng=rng, ensemble="copy_cycle")
    assert R.boundary_edges(sys, range(sys.k)) == tuple()
    assert len(R.boundary_edges(sys, (0, 1))) > 0


def test_raw_boundary_cancellation_is_z2_symmetric_difference():
    rng = np.random.default_rng(2)
    sys = R.make_closed_system(k=5, q=2, rng=rng, ensemble="copy_cycle")
    assert R.raw_boundary_cancellation_holds(sys, (0,), (1, 2))


def test_observer_copy_pair_boundary_measure_detects_live_binary_boundary():
    rng = np.random.default_rng(3)
    sys = R.make_closed_system(k=4, q=2, rng=rng, ensemble="observer_copy_pair")
    schedules = R.schedule_sample(sys.k, rng, max_schedules=None)
    sm = R.step_maps_for_schedules(sys, schedules)
    rec = R.recurrent_states_from_step_maps(sm)
    # For k=4, observer region A is vertices (2,3); B0 feeds A0.
    bm = R.boundary_measure(sys, (2, 3), horizon=1, step_maps=sm, recurrent_states=rec, schedules=schedules, rng=rng)
    assert bm.observer == (2, 3)
    assert len(bm.raw_boundary_edges) > 0
    assert bm.component_count == 2
    assert bm.binary_c2_like
    assert bm.zero_error_code_size >= 2




def test_observer_parity_pair_gives_c2_boundary_at_q4():
    rng = np.random.default_rng(33)
    sys = R.make_closed_system(k=4, q=4, rng=rng, ensemble="observer_parity_pair")
    schedules = R.schedule_sample(sys.k, rng, max_schedules=12)
    sm = R.step_maps_for_schedules(sys, schedules)
    rec = R.recurrent_states_from_step_maps(sm)
    bm = R.boundary_measure(sys, (2, 3), horizon=1, step_maps=sm, recurrent_states=rec, schedules=schedules, rng=rng, do_shannon=False)
    assert bm.component_count == 2
    assert bm.binary_c2_like

def test_gluing_measure_runs_on_disjoint_observers():
    rng = np.random.default_rng(4)
    sys = R.make_closed_system(k=5, q=2, rng=rng, ensemble="copy_cycle")
    schedules = R.schedule_sample(sys.k, rng, max_schedules=None)
    sm = R.step_maps_for_schedules(sys, schedules)
    rec = R.recurrent_states_from_step_maps(sm)
    gm = R.gluing_measure(sys, (0,), (1, 2), horizon=1, step_maps=sm, recurrent_states=rec, schedules=schedules, rng=rng)
    assert gm.raw_cancellation_ok
    assert gm.recurrent_state_count == len(rec)
    assert 0.0 <= gm.gluing_accuracy <= 1.0
    assert gm.residual_entropy_bits >= 0.0


def test_relative_geometry_sweep_small():
    bdf, gdf, summary = R.run_relative_geometry_sweep(
        k=4,
        q=2,
        ensembles=("observer_copy_pair", "constant"),
        n_instances=1,
        observer_min_size=1,
        observer_max_size=2,
        max_observers_per_instance=10,
        max_pairs_per_instance=10,
        horizon=1,
        base_seed=5,
        require_feedback=False,
        do_shannon=False,
        verbose=False,
    )
    assert not bdf.empty
    assert "verdict" in summary
    assert summary["full_system_boundary_empty"] is True
    assert set(["boundary_quotient_size", "raw_boundary_size"]).issubset(bdf.columns)
    assert set(["raw_cancellation_ok", "residual_entropy_bits"]).issubset(gdf.columns)
