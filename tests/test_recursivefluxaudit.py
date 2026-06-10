import pickle

import numpy as np

from relgauge.microlattice import make_microscopic_lattice
from relgauge.recursivefluxaudit import recursive_equalizer_rows, parity_features, audit_recursive_flux


def test_recursive_equalizer_rows_detects_nontrivial_relation():
    F = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=float)
    rows = recursive_equalizer_rows(F, patch_sizes=(1, 2, 4))
    assert rows
    assert any(r["shared_classes"] >= 2 for r in rows if r["patch_size"] == 1)
    pf = parity_features(F)
    assert pf["n_valid_plaquettes"] == 9
    assert pf["total_flux_parity"] == 0


def test_audit_recursive_flux_on_synthetic_winner(tmp_path):
    rng = np.random.default_rng(123)
    lat = make_microscopic_lattice(q=4, nx=2, ny=2, w=1, rng=rng, ensemble="c2link", p_flip=0.5)
    path = tmp_path / "winners.pkl"
    with open(path, "wb") as f:
        pickle.dump({"kind": "test", "winners": [{"fitness": 1.0, "metrics": {}, "lattice": lat}]}, f)
    df, cdf, tdf, summary = audit_recursive_flux(str(path), top_n=1, n_random_initial=128, patch_sizes=(1,2), verbose=False)
    assert len(df) == 1
    assert not cdf.empty
    assert summary["n_winners"] == 1
    assert "verdict" in summary
