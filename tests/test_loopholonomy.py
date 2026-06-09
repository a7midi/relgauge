from relgauge import loopholonomy as L


def test_holonomy_product_and_types():
    e = (0, 1, 2)
    swap01 = (1, 0, 2)
    swap12 = (0, 2, 1)
    h = L.holonomy_product([swap01, swap12])
    assert h != e
    assert L.perm_type(h) == "cycle_3"
    assert L.perm_order(h) == 3


def test_gauge_covariance_trial():
    gens = [(1, 0, 2), (0, 2, 1)]
    # generate_group is in transportalgebra, but run_loop_holonomy will use the same logic.
    from relgauge import transportalgebra as A
    group = sorted(A.generate_group(gens, n=3))
    edges = ((1, 0, 2), (0, 2, 1), (0, 1, 2), (2, 1, 0))
    import numpy as np
    rng = np.random.default_rng(123)
    for _ in range(20):
        out = L.gauge_covariance_trial(edges, group, rng)
        assert out["covariant"]
        assert out["same_conjugacy_type"]


def test_loop_holonomy_from_manual_maps(tmp_path):
    # Test the pure loop machinery without reading a pickle by monkeypatching map extraction.
    maps = [
        {"permutation": (0, 1, 2)},
        {"permutation": (1, 0, 2)},
        {"permutation": (0, 2, 1)},
        {"permutation": (2, 1, 0)},
    ]
    perms = L._choose_edge_maps(maps, use_unique=True)
    assert len(perms) == 4
    import numpy as np
    rng = np.random.default_rng(1)
    rows = []
    for edges in L._iter_loops(perms, 4, 0, rng):
        rows.append(L.holonomy_product(edges))
    assert len(set(rows)) == 6  # S3 is covered
