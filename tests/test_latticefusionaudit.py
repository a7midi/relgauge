import numpy as np

from relgauge import microlattice as ML
from relgauge import latticefusionaudit as LF


def test_pair_indices_random_excludes_self():
    rng = np.random.default_rng(0)
    pairs = LF._pair_indices(5, 20, rng, mode="random")
    assert len(pairs) == 20
    assert all(i != j for i, j in pairs)


def test_fuse_c2link_smoke_horizontal():
    rng = np.random.default_rng(1)
    a = ML.make_microscopic_lattice(q=3, nx=1, ny=1, w=1, rng=rng, ensemble="c2link", p_flip=0.0)
    b = ML.make_microscopic_lattice(q=3, nx=1, ny=1, w=1, rng=rng, ensemble="c2link", p_flip=0.0)
    fused, meta = LF.fuse_lattices(a, b, orientation="horizontal", rng=rng)
    assert fused.nx == 2 and fused.ny == 1
    assert meta["copy_right"]["skipped_overlap"] > 0
    row, _ = LF.evaluate_fusion(a, b, orientation="horizontal", rng=rng, initial_mode="source_all", n_random_initial=16)
    assert 0.0 <= row["z2_fraction"] <= 1.0
    assert row["seam_plaquette_count"] >= 1


def test_region_metrics_defined():
    rows = [dict(valid=True, flux=1.0, nontrivial=False), dict(valid=False, flux=float('nan'), nontrivial=False)]
    m = LF._region_metrics("x", rows)
    assert m["x_plaquette_count"] == 2
    assert 0.0 <= m["x_z2_fraction"] <= 1.0
