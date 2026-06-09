import pickle

import numpy as np

from relgauge import scdcdiamond as S
from relgauge import attractoraudit as A
from relgauge import blindselection as B


def test_functional_graph_decomposition_two_cycles():
    # 0 -> 1 -> 2 -> 1 and 3 <-> 4
    T = np.array([1, 2, 1, 4, 3], dtype=np.int64)
    fg = A.functional_graph_decomposition(T)
    assert len(fg.cycles) == 2
    assert sorted(len(c) for c in fg.cycles) == [2, 2]
    assert fg.distance_to_cycle[0] == 1
    assert fg.distance_to_cycle[1] == 0


def test_attractor_audit_runs_on_small_saved_winner(tmp_path):
    rng = np.random.default_rng(123)
    ds = S.make_scdc_diamond(kB=1, kC=1, kD=1, kA=2, q=2, w=1, rng=rng)
    path = tmp_path / "winners.pkl"
    B.save_winner_records(str(path), [{"diamond": ds, "metrics": {"blind_fitness": 0.0}}])
    df = A.run_attractor_audit(str(path), top_n=1, max_exact_states=1000, verbose=False)
    assert len(df) == 1
    assert "constant_basin_fraction" in df.columns
    summary = A.analyze_attractor_audit(df)
    assert "verdict" in summary
