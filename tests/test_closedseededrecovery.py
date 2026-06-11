import numpy as np

from relgauge import closedseededrecovery as CSR


def test_seeded_closed_recovery_small_run():
    df = CSR.run_seeded_closed_recovery(
        q=3,
        size=(2, 2),
        seed_ensembles=("copy", "random"),
        seed_mutation_rates=(0.0,),
        runs=1,
        population=4,
        generations=2,
        n_initial=128,
        horizon=4,
        schedule_samples=2,
        entry_rate=0.03,
        table_rate=0.03,
        verbose=False,
    )
    assert len(df) == 2 * 1 * 1 * 4 * 3
    assert "closed_c2_quotient" in df.columns
    summary = CSR.analyze_seeded_recovery(df)
    assert summary["n_rows"] == len(df)
    assert "verdict" in summary


def test_fitness_modes():
    row = dict(quotient_score=0.5, closed_c2_quotient=True, recurrent_flux_entropy_bits=2.0, n_plaquettes=4)
    assert CSR._fitness(row, "quotient") == 0.75
    assert CSR._fitness(row, "medium") > CSR._fitness(row, "quotient")
