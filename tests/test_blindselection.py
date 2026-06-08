import numpy as np
import pandas as pd

from relgauge import blindselection as BS


def test_clone_and_mutate_preserve_shapes():
    rng = np.random.default_rng(1)
    ds = BS.make_random_candidate(kB=2, kM=2, kA=2, q=3, w=1, rng=rng)
    child, info = BS.mutate_diamond(
        ds,
        rng,
        target="interface",
        proposal_mode="entry",
        entry_rate=1.0,
        table_rate=0.0,
    )
    assert child is not ds
    assert child.joint.k == ds.joint.k
    assert len(child.joint.rules) == len(ds.joint.rules)
    for r0, r1 in zip(ds.joint.rules, child.joint.rules):
        assert r0.shape == r1.shape
    assert info["entry_mutations"] > 0


def test_evaluate_candidate_returns_bounded_metrics():
    rng = np.random.default_rng(2)
    ds = BS.make_random_candidate(kB=2, kM=2, kA=2, q=3, w=1, rng=rng)
    row = BS.evaluate_candidate(ds, thresholds=(0.2,), quality_threshold=0.0)
    assert "blind_fitness" in row
    assert row["approx_residual_qcoords"] >= 0
    assert row["approx_residual_qcoords"] <= ds.w + 1e-9
    assert row["approx_quality_score"] >= 0
    assert "rule_interface_mi_norm_mean" in row


def test_run_null_sample_small():
    df = BS.run_null_sample(kB=2, kM=2, kA=2, w=1, q=3, n_samples=2, thresholds=(0.2,))
    assert len(df) == 2
    assert set(df.search_mode) == {"null"}
    res = BS.analyze_blind_selection(df)
    assert "verdict" in res
    assert res["blind_selection_bound_violations"] == 0


def test_run_evolution_small_smoke():
    df = BS.run_evolution(
        kB=2,
        kM=2,
        kA=2,
        w=1,
        q=3,
        n_runs=1,
        population=3,
        generations=1,
        thresholds=(0.2,),
        target="interface",
        proposal_mode="entry",
        entry_rate=0.2,
        table_rate=0.0,
        verbose=False,
    )
    assert len(df) == 6  # generation 0 and generation 1, three candidates each
    assert set(df.search_mode) == {"evolve"}
    assert sorted(df.generation.unique().tolist()) == [0, 1]


def test_evaluate_candidate_exact_stage_has_exact_metrics():
    rng = np.random.default_rng(7)
    ds = BS.make_random_candidate(kB=2, kM=2, kA=2, q=3, w=1, rng=rng)
    row = BS.evaluate_candidate(ds, thresholds=(0.2,), fitness_stage="exact", quality_threshold=0.0)
    assert "exact_residual_qcoords" in row
    assert "blind_fitness_stage" in row
    assert row["blind_fitness_stage"] == "exact"
    assert row["blind_primary_metric"] == "exact_residual_qcoords"
    assert 0 <= row["exact_residual_qcoords"] <= ds.w + 1e-9


def test_two_stage_evolution_switches_stage():
    df = BS.run_evolution(
        kB=2,
        kM=2,
        kA=2,
        w=1,
        q=3,
        n_runs=1,
        population=3,
        generations=2,
        pretrain_generations=1,
        fitness_mode="two_stage",
        thresholds=(0.2,),
        target="interface",
        proposal_mode="entry",
        entry_rate=0.2,
        table_rate=0.0,
        verbose=False,
    )
    assert set(df.search_mode) == {"evolve"}
    by_gen = {int(g): set(s.astype(str)) for g, s in df.groupby("generation")["blind_fitness_stage"]}
    assert by_gen[0] == {"approx"}
    assert by_gen[1] == {"exact"}
    assert by_gen[2] == {"exact"}


def test_exact_fitness_reports_exact_metrics():
    rng = np.random.default_rng(3)
    ds = BS.make_random_candidate(kB=2, kM=2, kA=2, q=4, w=1, rng=rng)
    row = BS.evaluate_candidate(ds, thresholds=(0.45,), fitness_stage="exact", quality_threshold=0.0)
    assert "exact_residual_qcoords" in row
    assert "blind_residual_qcoords" in row
    assert row["blind_fitness_stage"] == "exact"
    assert row["blind_residual_qcoords"] == row["exact_residual_qcoords"]
    assert 0 <= row["exact_residual_qcoords"] <= ds.w + 1e-9


def test_two_stage_switches_to_exact_stage():
    df = BS.run_evolution(
        kB=2,
        kM=2,
        kA=2,
        w=1,
        q=3,
        n_runs=1,
        population=3,
        generations=2,
        pretrain_generations=1,
        fitness_mode="two_stage",
        thresholds=(0.2,),
        target="interface",
        proposal_mode="entry",
        entry_rate=0.2,
        table_rate=0.0,
        verbose=False,
    )
    stage_by_gen = {int(g): set(s.astype(str)) for g, s in df.groupby("generation")["blind_fitness_stage"]}
    assert stage_by_gen[0] == {"approx"}
    assert stage_by_gen[1] == {"exact"}
    assert stage_by_gen[2] == {"exact"}


def test_run_blind_selection_two_stage_small():
    df = BS.run_blind_selection(
        ks_B=(2,),
        ks_M=(2,),
        ks_A=(2,),
        ws=(1,),
        q=3,
        null_samples=1,
        n_runs=1,
        population=3,
        generations=1,
        fitness_mode="two_stage",
        pretrain_generations=1,
        thresholds=(0.2,),
        target="interface",
        proposal_mode="entry",
        entry_rate=0.2,
        table_rate=0.0,
        verbose=False,
    )
    assert not df.empty
    assert "exact_residual_qcoords" in df.columns
    assert set(df["fitness_mode"].dropna().unique()) <= {"two_stage"}
    res = BS.analyze_blind_selection(df)
    assert "verdict" in res


def test_save_and_load_winner_records_roundtrip(tmp_path):
    rng = np.random.default_rng(11)
    ds = BS.make_random_candidate(kB=2, kM=2, kA=2, q=3, w=1, rng=rng)
    metrics = {"search_mode": "evolve", "run": 0, "generation": 0, "candidate": 0, "exact_residual_qcoords": 0.5}
    rec = BS.make_winner_record(ds, metrics)
    path = tmp_path / "winners.pkl"
    BS.save_winner_records(str(path), [rec], summary={"ok": True})
    loaded = BS.load_winner_records(str(path))
    assert len(loaded) == 1
    assert loaded[0]["format"] == "relgauge.blindwinner.v1"
    assert loaded[0]["diamond"].joint.k == ds.joint.k
