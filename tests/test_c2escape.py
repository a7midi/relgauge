import pickle

import numpy as np

from relgauge import c2escape as CE
from relgauge import blindsharedsquare as BS
from relgauge import sharedsquareholonomy as SH


def _make_seed_file(tmp_path):
    rng = np.random.default_rng(123)
    # block gives a valid flat binary transported square, a convenient C2-like seed
    sq = SH.make_shared_square(4, 1, rng, ensemble="block", mutation_rate=0.0)
    row = BS.evaluate_candidate(sq, rng, initial_mode="source_all", n_random_initial=64, fitness="transport")
    row["min_shared_classes"] = 2
    p = tmp_path / "seeds.pkl"
    with open(p, "wb") as f:
        pickle.dump({"kind": "blind_shared_square_winners", "winners": [{"fitness": row["blind_fitness"], "metrics": row, "square": sq}]}, f)
    return str(p)


def test_load_seed_winners(tmp_path):
    path = _make_seed_file(tmp_path)
    seeds = CE._load_seed_winners(path, top_n=1)
    assert len(seeds) == 1
    assert seeds[0][1].q == 4


def test_run_c2_escape_smoke(tmp_path):
    path = _make_seed_file(tmp_path)
    df, winners = CE.run_c2_escape(
        path,
        top_n=1,
        runs=1,
        population=3,
        generations=1,
        initial_mode="source_all",
        n_random_initial=64,
        seed_jitter_entry_rate=0.0,
        seed_jitter_table_rate=0.0,
        verbose=False,
    )
    assert len(df) > 0
    assert "min_shared_classes" in df.columns
    summary = CE.analyze_c2_escape(df)
    assert "verdict" in summary
    assert winners
