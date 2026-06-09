import numpy as np

from relgauge import blindsharedsquare as BSS
from relgauge import sharedsquareholonomy as SH


def test_evaluate_random_candidate_has_expected_fields():
    rng = np.random.default_rng(123)
    sq = BSS.random_square(3, 1, rng)
    row = BSS.evaluate_candidate(sq, rng, initial_mode="source_all", n_random_initial=64, fitness="transport")
    assert "blind_fitness" in row
    assert "min_residual_qcoords" in row
    assert "valid_square" in row


def test_mutation_preserves_shape():
    rng = np.random.default_rng(2)
    sq = SH.make_shared_square(3, 1, rng, ensemble="copy")
    mut = BSS.mutate_square(sq, rng, entry_rate=1.0, table_rate=0.2)
    assert mut.joint.k == sq.joint.k
    for r0, r1 in zip(sq.joint.rules, mut.joint.rules):
        assert r0.shape == r1.shape


def test_small_search_runs():
    df, winners = BSS.run_blind_shared_square(
        q=3,
        ws=(1,),
        null_samples=2,
        runs=1,
        population=3,
        generations=1,
        fitness="two_stage",
        pretrain_generations=1,
        initial_mode="source_all",
        n_random_initial=32,
        verbose=False,
    )
    assert len(df) > 0
    s = BSS.analyze_blind_shared_square(df)
    assert "verdict" in s
    assert len(winners) > 0
