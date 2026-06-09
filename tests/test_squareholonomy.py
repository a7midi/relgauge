import numpy as np

from relgauge import loopholonomy as H
from relgauge import squareholonomy as S


def test_path_delta_identity_for_equal_paths():
    # top = bottom => delta identity
    a = (1, 0, 2)
    b = (0, 2, 1)
    d = S._path_delta(a, b, a, b)["delta"]
    assert d == H.identity(3)


def test_gauge_covariance_square():
    # A simple S3 plaquette with nontrivial delta should transform by conjugacy.
    ab = (1, 0, 2)
    bd = (0, 2, 1)
    ac = (0, 1, 2)
    cd = (2, 1, 0)
    group = sorted({H.identity(3), ab, bd, cd})
    # close group using transportalgebra helper indirectly through square measure not needed here
    from relgauge import transportalgebra as A
    group = sorted(A.generate_group([ab, bd, cd], 3))
    rng = np.random.default_rng(123)
    for _ in range(10):
        r = S._gauge_covariance_square(ab, bd, ac, cd, group, rng)
        assert r["covariant"]
        assert r["same_conjugacy_type"]


def test_make_square_edges_smoke():
    rng = np.random.default_rng(1)
    edges = S.make_square_edges(2, 2, 2, 3, 1, rng, edge_ensemble="copy", mutation_rate=0)
    assert set(edges) == {"AB", "BD", "AC", "CD"}
    row = S.measure_square_instance(edges, initial_mode="source_all", n_random_initial=64)
    assert "path_holonomy_valid" in row
