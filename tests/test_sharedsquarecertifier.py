import numpy as np

from relgauge import sharedsquareholonomy as SH
from relgauge import sharedsquarecertifier as SC


def test_hidden_overwrite_certificate_for_generated_square():
    rng = np.random.default_rng(1)
    sq = SH.make_shared_square(4, 1, rng, ensemble="copy")
    cert = SC.hidden_overwrite_certificate(sq)
    assert cert["hidden_overwrite_certified"] is True
    assert cert["source_is_exactly_zero_pred"] is True
    assert cert["schedule_respects_feedforward"] is True


def test_certify_copy_square_valid_and_hidden_independent():
    rng = np.random.default_rng(2)
    sq = SH.make_shared_square(4, 1, rng, ensemble="copy")
    row = SC.certify_square(sq, rng=rng, hidden_check_samples=32)
    assert row["certified_valid_square"] is True
    assert row["hidden_output_independence"] is True
    assert row["certified_min_classes"] == 4
    assert row["generated_group_name"] == "trivial"
