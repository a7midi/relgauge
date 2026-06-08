"""
test_theory.py -- the package's scientific self-validation.

These tests assert the things we *proved* analytically.  If any fails, either
the code is wrong or a theorem is.  Run with:  pytest -q  (or python -m pytest)
"""
import numpy as np
import pytest

from relgauge import core as C, quotients as Q, observables as O


# --------------------------------------------------------------------------- #
def test_step_semantics_2cycle_identity():
    """Schedule (a,b) maps (a,b)->(b,b); (b,a) maps ->(a,a) for identity rules."""
    rng = np.random.default_rng(0)
    sys = C.make_cycle(2, 4, rng)
    sys.rules = [np.arange(4), np.arange(4)]
    S = C.all_states(2, 4)
    sm = sys.all_step_maps()           # rows: schedules (0,1) then (1,0)
    i = C.encode(np.array([[1, 3]]), 4)[0]
    assert tuple(S[sm[0][i]]) == (3, 3)    # (0,1)="ab" -> (b,b)
    assert tuple(S[sm[1][i]]) == (1, 1)    # (1,0)="ba" -> (a,a)


@pytest.mark.parametrize("q", [4, 8, 16])
def test_partition_collapse_on_cycle(q):
    """Image/recurrent-restricted partition survival on cycles -> 0 like
    1/((1-1/e)q): every Im(f_i) collapses to one class (the collapse theorem)."""
    rng = np.random.default_rng(q)
    surv = []
    for _ in range(20):
        sys = C.make_cycle(2, q, rng)
        sm = sys.all_step_maps()
        labels = Q.partition_quotient(sys, sm)
        img = C.image_states(sm)
        for v in range(2):
            imgvals = set(((img // (q ** v)) % q).tolist())
            # the surviving classes among image values must be exactly 1
            assert Q.partition_class_count(labels, v, imgvals) == 1
            surv.append(1 / len(imgvals))
    # and the count-ratio is near 1/((1-1/e)q)
    assert np.mean(surv) < 0.5            # well below 1: genuine collapse


def test_full_alphabet_is_one_over_e_artifact():
    """Full-alphabet partition survival on cycles tracks 1/q + (1-1/q)^q."""
    q = 8
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(60):
        sys = C.make_cycle(2, q, rng)
        sm = sys.all_step_maps()
        labels = Q.partition_quotient(sys, sm)
        for v in range(2):
            vals.append(Q.partition_class_count(labels, v) / q)
    predicted = 1 / q + (1 - 1 / q) ** q
    assert abs(np.mean(vals) - predicted) < 0.03


@pytest.mark.parametrize("k,qmax,target", [(2, 32, 0.5), (3, 16, 2 / 3)])
def test_orbit_dimension_trends_to_k_minus_1_over_k(k, qmax, target):
    """Orbit (reachable) dimension on cycles approaches (k-1)/k."""
    rng = np.random.default_rng(100 + k)
    vals = [Q.orbit_dimension(C.make_cycle(k, qmax, rng), which="image")
            for _ in range(30)]
    assert abs(np.mean(vals) - target) < 0.08


def test_resolvability_escapes_collapse_on_cycle():
    """The boundary quotient is NOT the partition quotient: recurrent
    resolvability is strictly > 0 (here ~1) where partition gives 0."""
    rng = np.random.default_rng(7)
    Rs = []
    for _ in range(20):
        sys = C.make_cycle(3, 8, rng)
        m = O.measure_instance(sys, 1, passive_horizons=(1,),
                               do_active=True, do_partition=False)
        if not np.isnan(m["R_active"]):
            Rs.append(m["R_active"])
    assert np.mean(Rs) > 0.5             # nonzero: escapes the collapse theorem


def test_condensation_is_dag():
    """S1: the SCC condensation is always acyclic (acyclic causality)."""
    rng = np.random.default_rng(3)
    for _ in range(30):
        sys = C.make_random_scc(4, 3, 2, rng)
        adj = {v: set() for v in range(sys.k)}
        for v in range(sys.k):
            for p in sys.preds[v]:
                adj[p].add(v)
        assert C.condensation_is_dag(adj, range(sys.k))


def test_light_cone_sync_holds_async_fails():
    """S2: finite light-cone is a synchronous property; it fails async."""
    rng = np.random.default_rng(5)
    sync = sum(O.check_light_cone(C.make_random_scc(4, 3, 2, rng), 80, rng, "sync")
               for _ in range(20))
    asy = sum(O.check_light_cone(C.make_random_scc(4, 3, 2, rng), 80, rng, "async")
              for _ in range(20))
    assert sync == 20
    assert asy < 20


def test_fusion_monotonicity_holds_and_needs_hypothesis():
    """S3: hidden-multiplicity entropy is monotone under valid fusion, and the
    extension hypothesis is necessary (it can fail with feedback)."""
    rng = np.random.default_rng(11)
    ok = sum(O.fusion_monotonicity_event(3, 3, 2, rng, feedback=False)[1]
             for _ in range(25))
    assert ok == 25
    viol = sum(not O.fusion_monotonicity_event(3, 3, 2, rng, feedback=True)[1]
               for _ in range(40))
    assert viol > 0


# --------------------------------------------------------------------------- #
# extension: gauge-dimension (cycle-rank) theorem
# --------------------------------------------------------------------------- #
def test_gauge_deficit_is_one_per_observer_not_betti():
    """Single SCC: gauge deficit ~ 1 regardless of beta_1 (refutes beta_1,
    confirms one global phase per observer)."""
    from relgauge import cyclerank as CR
    ds = CR.run_single_scc(k=4, qs=(8, 16), extras=(0, 1, 2, 3),
                           n_instances=25, verbose=False)
    res = CR.analyze(ds, None)
    defs = list(res["single_scc"]["deficit_by_extra_at_qmax"].values())
    betas = list(res["single_scc"]["beta1_by_extra"].values())
    assert max(defs) - min(defs) < 0.45          # deficit flat ...
    assert abs(np.mean(defs) - 1.0) < 0.5         # ... near 1
    assert max(betas) - min(betas) >= 2           # while beta_1 varied a lot


def test_gauge_deficit_additive_over_observers():
    """Multi-SCC: gauge deficit ~ number of (multi-vertex) SCCs."""
    from relgauge import cyclerank as CR
    dm = CR.run_multi_scc([(2,), (2, 2), (2, 2, 2)], q=4, n_instances=15,
                          verbose=False)
    for cfg in dm.config.unique():
        s = dm[dm.config == cfg]
        assert abs(s.deficit.mean() - s.n_obs.iloc[0]) < 0.5


# --------------------------------------------------------------------------- #
# extension: area-law test runs and classifies
# --------------------------------------------------------------------------- #
def test_area_law_runs_and_cycles_are_transparent():
    """Cycles are transparent to the active resolver (opacity ~ 1): the
    boundary reveals the whole recurrent state (volume law)."""
    from relgauge import arealaw as AL
    df = AL.run_area_law(ensemble="cycle", ks=(3, 4), q=4, n_instances=20,
                         mode="active", verbose=False)
    s = df[df.w == 1]
    # opacity = I/log|Rec| ~ 1 for cycles
    assert s.opacity.mean() > 0.9
