import numpy as np

from relgauge import diamond as D


def test_diamond_has_four_feedback_observers():
    rng = np.random.default_rng(0)
    ds = D.make_diamond_system(2, 2, 2, 2, 3, 1, rng)
    assert ds.k_total == 8
    assert D.nontrivial_scc_count(ds.joint) == 4
    assert ds.interface_BC and ds.interface_BD and ds.interface_CA and ds.interface_DA


def test_admissible_diamond_schedules_respect_blocks():
    rng = np.random.default_rng(1)
    ds = D.make_diamond_system(2, 2, 2, 2, 3, 1, rng)
    infos = D.admissible_diamond_schedules(ds)
    assert len(infos) == 32  # 2!*2!*2!*2!*2 branch orders
    for info in infos:
        pos = {v: i for i, v in enumerate(info.schedule)}
        # B before every C/D/A vertex; A after C and D.
        for b in range(ds.offB, ds.offC):
            for x in range(ds.offC, ds.offA + ds.kA):
                assert pos[b] < pos[x]
        for x in range(ds.offC, ds.offA):
            for a in range(ds.offA, ds.offA + ds.kA):
                assert pos[x] < pos[a]
        assert info.branch_order in {"CD", "DC"}


def test_diamond_herald_matrix_bounds():
    rng = np.random.default_rng(2)
    ds = D.make_diamond_system(2, 2, 2, 2, 3, 1, rng)
    sm, infos = D.step_maps_diamond(ds)
    ch = D.diamond_herald_matrix(
        ds,
        horizon=1,
        phase_mode="erased_value",
        protocol="first",
        step_maps=sm,
        infos=infos,
    )
    assert ch.P_herald.shape == (3, 3)
    assert np.all(np.isfinite(ch.P_herald))
    assert np.all(ch.P_herald >= -1e-12)
    assert np.all(ch.P_herald <= 1 + 1e-12)
    m = D._posterior_metrics(ch.P_herald)
    assert 0 <= m["herald_efficiency"] <= 1
    assert m["relation_mi_bits"] >= -1e-12


def test_run_diamond_sweep_smoke():
    df = D.run_diamond_sweep(
        ks_B=(2,),
        ks_C=(2,),
        ks_D=(2,),
        ks_A=(2,),
        ws=(1,),
        horizons=(1,),
        q=3,
        n_instances=1,
        phase_modes=("cut", "erased_value"),
        protocols=("first",),
        verbose=False,
    )
    assert len(df) == 2
    assert set(df["phase_mode"]) == {"cut", "erased_value"}
    res = D.analyze_diamond(df)
    assert "verdict" in res
    assert res["n_rows"] == 2
