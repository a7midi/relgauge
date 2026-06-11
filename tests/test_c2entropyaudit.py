import numpy as np
import pandas as pd

from relgauge import c2entropyaudit as CE
from relgauge import sharedsquareholonomy as SH


def test_conditional_stats_binary_pool():
    df = pd.DataFrame([
        {"valid_square": True, "binary_classes": True, "ternary_or_more": False, "full_classes": False, "min_classes": 2, "generated_group_name": "C2", "delta_type": "cycle_2"},
        {"valid_square": True, "binary_classes": True, "ternary_or_more": False, "full_classes": False, "min_classes": 2, "generated_group_name": "C2", "delta_type": "identity"},
        {"valid_square": False, "binary_classes": False, "ternary_or_more": False, "full_classes": False, "min_classes": 0, "generated_group_name": "invalid", "delta_type": "invalid"},
    ])
    st = CE._conditional_stats(df)
    assert st["valid_count"] == 2
    assert st["binary_given_valid"] == 1.0
    assert st["min_classes_counts"] == {"2": 2}


def test_row_from_structured_candidate_runs():
    rng = np.random.default_rng(3)
    sq = SH.make_shared_square(4, 1, rng, ensemble="block")
    row = CE._row_from_candidate(sq, rng, "test", 4, 1, "source_all", 64)
    assert row["min_classes"] >= 2
    assert "valid_square" in row
