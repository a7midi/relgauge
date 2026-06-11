import numpy as np

from relgauge import observerboundarygeometry as OBG
from relgauge import observerrelativeholonomy as ORH


def test_edge_common_factor_detects_binary_copy_channel():
    q = 4
    # Two SCC observers {0,1} and {2,3}; edge 1 -> 2 copies value mod binary partition.
    preds = [(1,), (0,), (1,), (2,)]
    tables = []
    # 0 copies 1, 1 copies 0, 2 copies 1, 3 copies 2.
    for _ in preds:
        tables.append(np.arange(q, dtype=np.int16))
    sys = OBG.FiniteRelationalSystem(q=q, preds=[tuple(p) for p in preds], tables=tables)
    rng = np.random.default_rng(1)
    eq = ORH.edge_common_factor(sys, (0, 1), [(1, 2)], 256, 256, rng)
    assert eq.n_classes == q  # full copy has q exact classes
    assert eq.live
    assert eq.transport_mi_norm == 1.0


def test_orient_cycle_as_two_paths_on_diamond():
    cycle = (0, 1, 3, 2)
    directed = {(0, 1), (1, 3), (0, 2), (2, 3)}
    oriented = ORH.orient_cycle_as_two_paths(cycle, directed)
    assert oriented is not None
    source, sink, paths = oriented
    assert source == 0 and sink == 3
    assert sorted(paths) == sorted([[0, 1, 3], [0, 2, 3]])


def test_small_observer_relative_holonomy_run_smokes():
    edge_df, hol_df, summary = ORH.run_audit(
        q=3, graphs=2, vertices=10, graph_mode="componented", components=4,
        edge_prob=0.2, inter_prob=0.8, extra_intra_prob=0.3, max_pred=4,
        capacity_threshold=0.0, cycle_capacity_threshold=0.0,
        max_channel_inputs=64, max_channel_backgrounds=64, max_local_configs=128,
        cycle_max_len=5, seed=123, verbose=False,
    )
    assert "verdict" in summary
    assert summary["full_boundary_violations"] == 0
    assert len(edge_df) >= 0 and len(hol_df) >= 0
