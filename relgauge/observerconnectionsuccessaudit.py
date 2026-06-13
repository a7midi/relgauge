"""
observerconnectionsuccessaudit.py

Multi-seed success audit for blind observer-relative connection selection.

This module repeatedly calls blindobserverconnection.evolve_observer_connection
with different RNG seeds and records how often the consistency-only v4 search
finds:

  * true multi-port observer frames,
  * true frame transports,
  * completed two-branch observer diamonds,
  * automorphism-valid loop closure,
  * flat versus nontrivial holonomy, audited post-hoc.

It does not alter the selection objective.  It is a statistical wrapper for the
same purity-preserving score used in blindobserverconnection.py.

CLI example
-----------
python -m relgauge.observerconnectionsuccessaudit 4 ^
  --runs 30 --seed-start 0 ^
  --graph-mode frame_random_diamond --seed-mode random ^
  --population 48 --generations 120 --elite 6 ^
  --mutation-rate 0.08 --random-injection 0.10 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/blind_observer_connection_v4_multiseed.csv ^
  --winners example_results/blind_observer_connection_v4_multiseed_winners.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import blindobserverconnection as BOC
except Exception:  # pragma: no cover
    import blindobserverconnection as BOC  # type: ignore


def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def first_generation(hist_df, predicate) -> Optional[int]:
    if hist_df is None or len(hist_df) == 0:
        return None
    for _, row in hist_df.iterrows():
        try:
            if bool(predicate(row)):
                return int(row["generation"])
        except Exception:
            continue
    return None


def _bool_int(x) -> int:
    return int(bool(x))


def summarize_one_run(run_id: int, seed: int, hist_df, summary: Dict) -> Dict:
    final = summary.get("final_best_summary", {}) or {}
    all_time = summary.get("all_time_best_summary", {}) or {}

    def col(name: str, default=0):
        return all_time.get(name, final.get(name, default))

    row = dict(
        run_id=int(run_id),
        seed=int(seed),
        verdict=str(summary.get("verdict", "")),
        final_best_score=float(summary.get("final_best_score", 0.0)),
        all_time_best_score=float(summary.get("all_time_best_score", 0.0)),
        final_found_connection=_bool_int(summary.get("final_found_connection", False)),
        final_found_flat_connection=_bool_int(summary.get("final_found_flat_connection", False)),
        final_found_automorphism_valid_connection=_bool_int(summary.get("final_found_automorphism_valid_connection", False)),
        final_found_nontrivial_holonomy=_bool_int(summary.get("final_found_nontrivial_holonomy", False)),
        final_found_global_nontrivial_holonomy=_bool_int(summary.get("final_found_global_nontrivial_holonomy", False)),
        all_time_found_connection=_bool_int(summary.get("all_time_found_connection", False)),
        all_time_found_flat_connection=_bool_int(summary.get("all_time_found_flat_connection", False)),
        all_time_found_automorphism_valid_connection=_bool_int(summary.get("all_time_found_automorphism_valid_connection", False)),
        all_time_found_nontrivial_holonomy=_bool_int(summary.get("all_time_found_nontrivial_holonomy", False)),
        all_time_found_global_nontrivial_holonomy=_bool_int(summary.get("all_time_found_global_nontrivial_holonomy", False)),
        all_time_path_agreement=float(summary.get("all_time_path_agreement", col("max_path_agreement", 0.0))),
        all_time_loop_automorphism_validity=float(summary.get("all_time_loop_automorphism_validity", col("max_loop_automorphism_validity", 0.0))),
        all_time_two_branch_completion=float(summary.get("all_time_two_branch_completion", col("max_two_branch_completion", 0.0))),
        all_time_true_live_frame_fraction=float(all_time.get("true_live_frame_fraction", 0.0)),
        all_time_true_live_frame_transport_fraction=float(all_time.get("true_live_frame_transport_fraction", 0.0)),
        all_time_nontrivial_holonomy_count=int(all_time.get("n_nontrivial_holonomy", 0)),
        all_time_global_nontrivial_holonomy_count=int(all_time.get("global_nontrivial_holonomy", 0)),
        all_time_global_generated_group=str(all_time.get("global_generated_group", "none")),
        all_time_global_holonomy_type_counts=json.dumps(_jsonable(all_time.get("global_holonomy_type_counts", {})), sort_keys=True),
        first_true_frames=first_generation(hist_df, lambda r: float(r.get("best_true_live_frame_fraction", 0.0)) >= 1.0),
        first_true_transports=first_generation(hist_df, lambda r: float(r.get("best_true_live_frame_transport_fraction", 0.0)) >= 1.0),
        first_two_branch=first_generation(hist_df, lambda r: float(r.get("best_max_two_branch_completion", 0.0)) >= 1.0),
        first_loop_auto=first_generation(hist_df, lambda r: float(r.get("best_max_loop_automorphism_validity", 0.0)) >= 1.0 or int(r.get("best_global_valid_holonomy", 0)) > 0),
        first_flat=first_generation(hist_df, lambda r: int(r.get("best_flat_connections", 0)) > 0 or float(r.get("best_max_path_agreement", 0.0)) >= 1.0),
        first_nontrivial=first_generation(hist_df, lambda r: int(r.get("best_nontrivial_holonomy", 0)) > 0 or int(r.get("best_global_nontrivial_holonomy", 0)) > 0),
    )
    row["all_time_found_any_nontrivial"] = int(row["all_time_found_nontrivial_holonomy"] or row["all_time_found_global_nontrivial_holonomy"])
    row["final_found_any_nontrivial"] = int(row["final_found_nontrivial_holonomy"] or row["final_found_global_nontrivial_holonomy"])
    row["all_time_flat_only"] = int(row["all_time_found_automorphism_valid_connection"] and not row["all_time_found_any_nontrivial"] and row["all_time_found_flat_connection"])
    return row


def aggregate_seed_rows(seed_df) -> Dict:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas required")
    n = int(len(seed_df))
    if n == 0:
        return dict(verdict="NO RUNS", n_runs=0)

    def mean_bool(col: str) -> float:
        return float(seed_df[col].astype(float).mean()) if col in seed_df else 0.0

    def median_existing(col: str) -> Optional[float]:
        if col not in seed_df:
            return None
        vals = seed_df[col].dropna().astype(float)
        if len(vals) == 0:
            return None
        return float(vals.median())

    group_counts = {str(k): int(v) for k, v in Counter(seed_df.get("all_time_global_generated_group", [])).items()}
    verdict_counts = {str(k): int(v) for k, v in Counter(seed_df.get("verdict", [])).items()}
    nontriv_rate = mean_bool("all_time_found_any_nontrivial")
    auto_rate = mean_bool("all_time_found_automorphism_valid_connection")
    if nontriv_rate > 0.0:
        verdict = "MULTI-SEED AUDIT: NONTRIVIAL OBSERVER-FRAME HOLONOMY FOUND IN RANDOM-START RUNS"
    elif auto_rate > 0.0:
        verdict = "MULTI-SEED AUDIT: AUTOMORPHISM-VALID CONNECTIONS FOUND, HOLONOMY FLAT IN THIS SAMPLE"
    else:
        verdict = "MULTI-SEED AUDIT: NO AUTOMORPHISM-VALID CONNECTIONS FOUND IN THIS SAMPLE"
    return dict(
        verdict=verdict,
        n_runs=n,
        final_connection_fraction=mean_bool("final_found_automorphism_valid_connection"),
        all_time_connection_fraction=auto_rate,
        final_nontrivial_fraction=mean_bool("final_found_any_nontrivial"),
        all_time_nontrivial_fraction=nontriv_rate,
        all_time_flat_only_fraction=mean_bool("all_time_flat_only"),
        final_flat_fraction=mean_bool("final_found_flat_connection"),
        all_time_flat_fraction=mean_bool("all_time_found_flat_connection"),
        mean_final_best_score=float(seed_df["final_best_score"].mean()) if "final_best_score" in seed_df else 0.0,
        mean_all_time_best_score=float(seed_df["all_time_best_score"].mean()) if "all_time_best_score" in seed_df else 0.0,
        median_first_true_frames=median_existing("first_true_frames"),
        median_first_true_transports=median_existing("first_true_transports"),
        median_first_two_branch=median_existing("first_two_branch"),
        median_first_loop_auto=median_existing("first_loop_auto"),
        median_first_flat=median_existing("first_flat"),
        median_first_nontrivial=median_existing("first_nontrivial"),
        group_counts=group_counts,
        verdict_counts=verdict_counts,
    )


def run_success_audit(
    q: int = 4,
    runs: int = 10,
    seed_start: int = 0,
    seed_stride: int = 1,
    graph_mode: str = "frame_random_diamond",
    vertices: int = 8,
    components: int = 5,
    edge_prob: float = 0.16,
    inter_prob: float = 0.35,
    extra_intra_prob: float = 0.25,
    max_pred: Optional[int] = 4,
    seed_mode: str = "random",
    population: int = 48,
    generations: int = 120,
    elite: int = 6,
    mutation_rate: float = 0.08,
    min_mutations: int = 1,
    max_mutations: Optional[int] = None,
    random_injection: float = 0.10,
    max_channel_inputs: int = 256,
    max_channel_backgrounds: int = 256,
    max_state_samples: int = 2048,
    state_warmup: int = 4,
    cycle_max_len: int = 6,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    cycle_objective: str = "best",
    verbose: bool = True,
) -> Tuple[object, object, List[Dict], Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for success audit")
    seed_rows: List[Dict] = []
    hist_rows: List[object] = []
    winner_payload: List[Dict] = []
    run_summaries: List[Dict] = []

    for run_id in range(int(runs)):
        seed = int(seed_start + run_id * seed_stride)
        if verbose:
            print(f"success-audit run={run_id+1}/{runs} seed={seed}", flush=True)
        hist_df, winners, summary = BOC.evolve_observer_connection(
            q=q,
            graph_mode=graph_mode,
            vertices=vertices,
            components=components,
            edge_prob=edge_prob,
            inter_prob=inter_prob,
            extra_intra_prob=extra_intra_prob,
            max_pred=max_pred,
            seed_mode=seed_mode,
            population=population,
            generations=generations,
            elite=elite,
            mutation_rate=mutation_rate,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            random_injection=random_injection,
            max_channel_inputs=max_channel_inputs,
            max_channel_backgrounds=max_channel_backgrounds,
            max_state_samples=max_state_samples,
            state_warmup=state_warmup,
            cycle_max_len=cycle_max_len,
            min_frame_ports=min_frame_ports,
            frame_coordinate_mode=frame_coordinate_mode,
            cycle_objective=cycle_objective,
            seed=seed,
            verbose=False,
        )
        row = summarize_one_run(run_id, seed, hist_df, summary)
        seed_rows.append(row)
        run_summaries.append(_jsonable(summary))
        h = hist_df.copy()
        h.insert(0, "run_id", run_id)
        h.insert(1, "seed", seed)
        hist_rows.append(h)
        # Save the all-time best and final best for later certification.
        labels = ["all_time_best", "final_best"]
        for label, cand in zip(labels, winners[:2]):
            winner_payload.append(dict(
                run_id=run_id,
                seed=seed,
                role=label,
                system=cand.system,
                score=float(cand.score),
                summary=_jsonable(cand.summary),
                graph_id=int(cand.graph_id),
            ))
        if verbose:
            print(
                f"  score={row['all_time_best_score']:.4f} auto={row['all_time_found_automorphism_valid_connection']} "
                f"nontriv={row['all_time_found_any_nontrivial']} first_nontriv={row['first_nontrivial']}",
                flush=True,
            )

    seed_df = pd.DataFrame(seed_rows)
    hist_df_all = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()
    aggregate = aggregate_seed_rows(seed_df)
    aggregate.update(
        q=int(q),
        runs=int(runs),
        seed_start=int(seed_start),
        seed_stride=int(seed_stride),
        graph_mode=str(graph_mode),
        seed_mode=str(seed_mode),
        population=int(population),
        generations=int(generations),
        elite=int(elite),
        mutation_rate=float(mutation_rate),
        random_injection=float(random_injection),
        min_frame_ports=int(min_frame_ports),
        frame_coordinate_mode=str(frame_coordinate_mode),
        cycle_objective=str(cycle_objective),
    )
    summary = dict(aggregate=_jsonable(aggregate), run_summaries=run_summaries)
    return seed_df, hist_df_all, winner_payload, summary


def write_outputs(seed_df, hist_df, winners: List[Dict], summary: Dict, out: str, winners_path: Optional[str]) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    seed_df.to_csv(out, index=False)
    hist_df.to_csv(derived_path(out, "_history"), index=False)
    with open(derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)
    if winners_path:
        os.makedirs(os.path.dirname(winners_path) or ".", exist_ok=True)
        with open(winners_path, "wb") as f:
            pickle.dump(winners, f)


def make_plot(seed_df, hist_df, summary: Dict, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(hist_df):
        grouped = hist_df.groupby("generation")
        mean_best = grouped["best_score"].mean()
        max_best = grouped["best_score"].max()
        ax.plot(mean_best.index, mean_best.values, label="mean best")
        ax.plot(max_best.index, max_best.values, label="max best")
    ax.set_title(summary.get("aggregate", {}).get("verdict", "observer connection success audit"), fontsize=10)
    ax.set_xlabel("generation")
    ax.set_ylabel("best consistency score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-seed success audit for blind observer connection selection")
    p.add_argument("q", type=int, nargs="?", default=4)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-stride", type=int, default=1)
    p.add_argument("--graph-mode", choices=["frame_random_diamond", "componented", "er", "frame_flat_c2", "frame_twist_c2", "frame_theta_flat_c2", "frame_theta_twist_c2", "frame_random_theta", "frame_double_flat_c2", "frame_double_twist_c2", "frame_random_double_diamond"], default="frame_random_diamond")
    p.add_argument("--vertices", type=int, default=8)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.16)
    p.add_argument("--inter-prob", type=float, default=0.35)
    p.add_argument("--extra-intra-prob", type=float, default=0.25)
    p.add_argument("--max-pred", type=int, default=4)
    p.add_argument("--seed-mode", choices=["random", "flat_c2", "twist_c2", "randomized_flat_topology", "diamond_random_rules", "theta_flat_c2", "theta_twist_c2", "theta_random_rules", "double_flat_c2", "double_twist_c2", "double_random_rules"], default="random")
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--generations", type=int, default=120)
    p.add_argument("--elite", type=int, default=6)
    p.add_argument("--mutation-rate", type=float, default=0.08)
    p.add_argument("--min-mutations", type=int, default=1)
    p.add_argument("--max-mutations", type=int, default=0)
    p.add_argument("--random-injection", type=float, default=0.10)
    p.add_argument("--max-channel-inputs", type=int, default=256)
    p.add_argument("--max-channel-backgrounds", type=int, default=256)
    p.add_argument("--max-state-samples", type=int, default=2048)
    p.add_argument("--state-warmup", type=int, default=4)
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--min-frame-ports", type=int, default=2)
    p.add_argument("--frame-coordinate-mode", choices=["local_charts", "common_factor"], default="local_charts")
    p.add_argument("--cycle-objective", choices=["best", "all"], default="best")
    p.add_argument("--out", default="example_results/blind_observer_connection_multiseed.csv")
    p.add_argument("--winners", default="example_results/blind_observer_connection_multiseed_winners.pkl")
    p.add_argument("--plot", default="example_results/fig_blind_observer_connection_multiseed.png")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    max_pred = None if args.max_pred is None or args.max_pred <= 0 else int(args.max_pred)
    max_mut = None if args.max_mutations is None or args.max_mutations <= 0 else int(args.max_mutations)
    seed_df, hist_df, winners, summary = run_success_audit(
        q=args.q,
        runs=args.runs,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        graph_mode=args.graph_mode,
        vertices=args.vertices,
        components=args.components,
        edge_prob=args.edge_prob,
        inter_prob=args.inter_prob,
        extra_intra_prob=args.extra_intra_prob,
        max_pred=max_pred,
        seed_mode=args.seed_mode,
        population=args.population,
        generations=args.generations,
        elite=args.elite,
        mutation_rate=args.mutation_rate,
        min_mutations=args.min_mutations,
        max_mutations=max_mut,
        random_injection=args.random_injection,
        max_channel_inputs=args.max_channel_inputs,
        max_channel_backgrounds=args.max_channel_backgrounds,
        max_state_samples=args.max_state_samples,
        state_warmup=args.state_warmup,
        cycle_max_len=args.cycle_max_len,
        min_frame_ports=args.min_frame_ports,
        frame_coordinate_mode=args.frame_coordinate_mode,
        cycle_objective=args.cycle_objective,
        verbose=not args.quiet,
    )
    write_outputs(seed_df, hist_df, winners, summary, args.out, args.winners)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        make_plot(seed_df, hist_df, summary, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(_jsonable(summary["aggregate"]), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {derived_path(args.out, '_history')}")
    print(f"wrote {derived_path(args.out, '_summary').replace('.csv', '.json')}")
    if args.winners:
        print(f"wrote {args.winners}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
