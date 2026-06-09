import numpy as np

from relgauge import basininvariantaudit as BIA


def test_mutual_info_identity():
    x = np.array([0, 0, 1, 1, 2, 2])
    mi = BIA._mutual_info_discrete(x, x)
    h = BIA._entropy_from_counts(np.bincount(x))
    assert abs(mi - h) < 1e-12


def test_encode_word():
    digits = np.array([[1, 0], [0, 1], [2, 2]])
    out = BIA._encode_word(digits, 3)
    assert out.tolist() == [1, 3, 8]
