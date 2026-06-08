from relgauge import algebraaudit as A
from relgauge import core as C
from relgauge import scdcdiamond as S
import numpy as np


def test_group_classification_c4_v4_s4():
    q = 4
    c4 = A.generated_group([(1, 2, 3, 0)], q)
    info = A.classify_group(c4, q)
    assert info["group_name"] == "C4"
    assert info["group_zq_like"] is True

    v4 = A.generated_group([(1, 0, 3, 2), (2, 3, 0, 1)], q)
    info = A.classify_group(v4, q)
    assert info["group_name"] == "V4"
    assert info["group_abelian"] is True
    assert info["group_cyclic"] is False

    s4 = A.generated_group([(1, 0, 2, 3), (1, 2, 3, 0)], q)
    info = A.classify_group(s4, q)
    assert info["group_name"] == "S4"
    assert info["group_order"] == 24


def test_restricted_functions_for_edge_detects_permutation_context():
    q = 3
    # One target vertex 1 reads source 0 and context vertex 2.
    preds = [(), (0, 2), ()]
    # rule for vertex 1: output = source + context mod q.
    table = []
    for ctx in range(q):
        pass
    rule = np.zeros(q ** 2, dtype=np.int64)
    for x0 in range(q):
        for x2 in range(q):
            rule[x0 + x2 * q] = (x0 + x2) % q
    rules = [np.array([0], dtype=np.int64), rule, np.array([0], dtype=np.int64)]
    joint = C.RelationalSystem(3, q, preds, rules)
    ds = S.SCDCDiamond(
        B=joint, Csys=joint, Dsys=joint, Askel=joint, joint=joint,
        kB=1, kC=1, kD=1, kA=0, q=q, w=1,
        offB=0, offC=1, offD=2, offA=3,
        interface_BC=((0, 1),), interface_BD=(), interface_CA=(), interface_DA=(), meta={},
    )
    rows = A.restricted_functions_for_edge(ds, (0, 1))
    assert len(rows) == q
    assert all(r["is_permutation"] for r in rows)
    group = A.generated_group([r["permutation"] for r in rows], q)
    info = A.classify_group(group, q)
    assert info["group_zq_like"] is True
