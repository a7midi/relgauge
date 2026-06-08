import numpy as np

from relgauge import blindselection as BS
from relgauge import commonfactoraudit as CFA
from relgauge import ruleselection as RS
from relgauge import scdcdiamond as S


def test_common_factor_audit_detects_copy_liveness(tmp_path):
    rng = np.random.default_rng(12)
    ds = S.make_scdc_diamond(kB=2, kC=2, kD=2, kA=2, q=4, w=1, rng=rng)
    RS.apply_rule_ensemble(ds, "copy", rng, mutation_rate=0.0)
    metrics = {"search_mode": "evolve", "run": 0, "generation": 0, "candidate": 0, "exact_residual_qcoords": 1.0}
    path = tmp_path / "copy_winner.pkl"
    BS.save_winner_records(str(path), [BS.make_winner_record(ds, metrics)], summary={})
    df = CFA.run_common_factor_audit(
        str(path),
        top_n=1,
        mutation_trials=1,
        mutation_rates=(0.0,),
        future_sample_cap=20000,
        verbose=False,
    )
    assert len(df) == 1
    assert df.iloc[0].exact_residual_qcoords >= 1.0 - 1e-9
    assert bool(df.iloc[0].live_common_factor)
    assert df.iloc[0].source_to_label_mi_over_label_entropy > 0.9
    res = CFA.analyze_common_factor_audit(df)
    assert "verdict" in res
