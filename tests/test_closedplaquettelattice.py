import numpy as np

from relgauge import closedplaquettelattice as CL


def test_closed_lattice_is_scc_and_measures():
    rng = np.random.default_rng(1)
    lat = CL.make_closed_lattice(q=4, rows=2, cols=2, rng=rng, ensemble="c2xor")
    assert lat.meta["closed_scc"] is True
    row = CL.measure_closed_lattice(lat, n_initial=256, horizon=6, schedule_samples=2, rng=rng)
    assert row["closed_scc"] is True
    assert row["link_bit_entropy_bits"] > 0
    assert 0 <= row["quotient_determinism_accuracy"] <= 1


def test_binary_partitions():
    parts = CL.binary_partitions(4)
    assert len(parts) == 7
    assert all(p[0] == 0 for p in parts)
    assert all(0 < p.sum() < 4 for p in parts)


def test_sweep_summary_runs():
    df = CL.run_closed_sweep(q=3, sizes=[(2,2)], ensembles=("c2xor", "random"), instances=1, n_initial=128, horizon=4, schedule_samples=2, verbose=False)
    summary = CL.analyze_closed(df)
    assert summary["n_rows"] == 2
    assert "verdict" in summary
