import numpy as np

from relgauge import observerboundarygeometry as OBG


def test_full_system_has_no_boundary_but_proper_subset_can():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    full_in, full_out = OBG.boundary_edges({0, 1, 2, 3}, edges)
    assert full_in == [] and full_out == []
    inc, out = OBG.boundary_edges({0, 1, 2}, edges)
    assert inc == []
    assert out == [(2, 3)]


def test_tarjan_finds_two_feedback_observers():
    edges = [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)]
    comps = [set(c) for c in OBG.tarjan_scc(4, edges)]
    assert {0, 1} in comps
    assert {2, 3} in comps


def test_componented_audit_detects_observer_rows_and_no_full_boundary_violation():
    rng = np.random.default_rng(123)
    sys = OBG.make_componented_system(q=3, n=9, components=3, inter_prob=0.9, extra_intra_prob=0.2, rng=rng, max_pred=4)
    obs, edges, cycles, summary = OBG.analyze_system(sys, graph_id=0, capacity_threshold=0.0, rng=rng)
    assert summary["full_boundary_violation"] == 0
    assert summary["n_feedback_scc"] >= 1
    assert len(obs) >= 1
    assert summary["n_observer_edges"] >= 1


def test_cycle_finder_on_diamond_observer_graph():
    nodes = [0, 1, 2, 3]
    undirected_edges = [(0, 1), (1, 3), (0, 2), (2, 3)]
    cycles = OBG.find_undirected_cycles(nodes, undirected_edges, max_len=4)
    assert (0, 1, 3, 2) in cycles or (0, 2, 3, 1) in cycles
