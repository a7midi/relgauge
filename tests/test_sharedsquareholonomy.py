import numpy as np

from relgauge import sharedsquareholonomy as S


def test_copy_shared_square_flat_valid():
    rng = np.random.default_rng(1)
    sq = S.make_shared_square(q=4, w=1, rng=rng, ensemble="copy", mutation_rate=0.0)
    m = S.measure_shared_square(sq, initial_mode="source_all", n_random_initial=64, rng=rng)
    assert m["valid_square"]
    assert not m["nontrivial_path_holonomy"]
    assert m["delta_type"] == "identity"


def test_permutive_shared_square_smoke():
    rng = np.random.default_rng(2)
    sq = S.make_shared_square(q=4, w=1, rng=rng, ensemble="permutive", mutation_rate=0.0)
    m = S.measure_shared_square(sq, initial_mode="source_all", n_random_initial=64, rng=rng)
    assert m["valid_square"]
    assert m["gauge_covariance_success"] is True
    assert m["conjugacy_class_success"] is True


def test_sweep_summary_runs():
    df = S.run_shared_square_sweep(q=3, ws=(1,), ensembles=("copy", "permutive", "random"), mutation_rates=(0.0,), instances=2, initial_mode="source_all", n_random_initial=64, verbose=False)
    summary = S.analyze_shared_square(df)
    assert summary["n_rows"] == 6
    assert "verdict" in summary
