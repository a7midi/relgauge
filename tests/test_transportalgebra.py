import numpy as np

from relgauge import temporalchain as T
from relgauge import transportalgebra as A


def test_permutation_group_classifier_c4():
    gen = (1, 2, 3, 0)
    g = A.generate_group([gen], n=4)
    c = A.classify_group(g, n=4)
    assert c["group_name"] == "C4"
    assert c["group_order"] == 4
    assert c["group_cyclic"] is True


def test_copy_temporal_chain_induced_transport_map_is_bijective():
    rng = np.random.default_rng(123)
    tc = T.make_rule_temporal_chain(kB=2, kM=2, kA=2, q=4, w=1, rng=rng, rule_ensemble="copy")
    labs = A.temporal_chain_labels(tc, initial_mode="source_all", n_random_initial=0, rng=rng)
    m = A.induced_transport_map(labs["source"], labs["s1"], labs["s2"], tc.q, tc.w)
    assert m["transport_mi_over_Hs2"] > 0.99
    assert m["source_to_s2_mi_over_Hs2"] > 0.99
    assert m["best_accuracy"] > 0.99
    assert m["is_bijection"] is True
