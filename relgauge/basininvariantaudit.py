"""
basininvariantaudit.py -- source-to-attractor-label audit for a selected winner.

This is a focused follow-up to attractoraudit.py.  The attractor audit can find
winners whose selected exact interface alphabet S is constant on attractor
basins.  This script asks whether that stable basin label is *live* with
respect to the source boundary:

    I(B_initial_boundary ; S_stable_at_fixed_point) / H(S_stable_at_fixed_point)

For a chosen saved blind-selection winner, it:
  1. reconstructs the exact selected common factor S from the original diamond
     convergence data;
  2. iterates every requested initial state under a deterministic schedule;
  3. records the initial B-boundary word and the stable S label at the reached
     fixed point;
  4. computes the normalized mutual information between them.

If the ratio is high, the attractor-stable label is a live basin invariant.  If
it is near zero, the stable basins exist but do not correlate with the source
boundary.

Winner pickle files are local artifacts.  Do not load pickle files from
untrusted sources.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Literal

import numpy as np
import pandas as pd

from . import blindselection as BS
from . import scdcdiamond as S
from . import commonfactoraudit as CFA
from . import attractoraudit as AA

InitialPool = S.InitialPool
ObserverInit = S.ObserverInit
SourceMode = Literal["pre", "post_b", "both"]


# --------------------------------------------------------------------------- #
# Basic information helpers
# --------------------------------------------------------------------------- #
def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    ok = (x >= 0) & (y >= 0)
    x = x[ok]
    y = y[ok]
    if x.shape != y.shape or len(x) == 0:
        return math.nan
    xv, xi = np.unique(x, return_inverse=True)
    yv, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xv), len(yv)), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)
    hx = _entropy_from_counts(joint.sum(axis=1))
    hy = _entropy_from_counts(joint.sum(axis=0))
    hxy = _entropy_from_counts(joint.reshape(-1))
    return float(hx + hy - hxy)


def _encode_word(digits: np.ndarray, q: int) -> np.ndarray:
    digits = np.asarray(digits, dtype=np.int64)
    if digits.ndim == 1:
        digits = digits.reshape(-1, 1)
    if digits.shape[1] == 0:
        return np.zeros(digits.shape[0], dtype=np.int64)
    powers = int(q) ** np.arange(digits.shape[1], dtype=np.int64)
    return (digits * powers).sum(axis=1).astype(np.int64)


def _word_digits(states: np.ndarray, q: int, vertices: tuple[int, ...]) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64)
    out = np.empty((len(states), len(vertices)), dtype=np.int64)
    for j, v in enumerate(vertices):
        out[:, j] = (states // (int(q) ** int(v))) % int(q)
    return out


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


# --------------------------------------------------------------------------- #
# State pools and source boundary words
# --------------------------------------------------------------------------- #
def _all_joint_states(ds: S.SCDCDiamond) -> np.ndarray:
    return np.arange(ds.n_joint_states, dtype=np.int64)


def _declared_initial_states(ds: S.SCDCDiamond, pool: InitialPool, observer_init: ObserverInit) -> np.ndarray:
    return S._initial_states(ds, pool, observer_init)


def _states_to_component_words(ds: S.SCDCDiamond, states: np.ndarray) -> dict[str, np.ndarray]:
    """Return encoded component-level words for optional diagnostics."""
    states = np.asarray(states, dtype=np.int64)
    return {
        "B_word": _encode_word(_word_digits(states, ds.q, tuple(range(ds.offB, ds.offC))), ds.q),
        "C_word": _encode_word(_word_digits(states, ds.q, tuple(range(ds.offC, ds.offD))), ds.q),
        "D_word": _encode_word(_word_digits(states, ds.q, tuple(range(ds.offD, ds.offA))), ds.q),
        "A_word": _encode_word(_word_digits(states, ds.q, tuple(range(ds.offA, ds.offA + ds.kA))), ds.q),
    }


def pre_b_boundary_words(ds: S.SCDCDiamond, states: np.ndarray) -> np.ndarray:
    """Initial B boundary word before the tick begins."""
    b_vertices = tuple(src for src, _ in ds.interface_BC[: ds.w])
    return _encode_word(_word_digits(states, ds.q, b_vertices), ds.q)


def post_b_boundary_words(ds: S.SCDCDiamond, states: np.ndarray, schedule: tuple[int, ...]) -> np.ndarray:
    """B boundary word after the B-prefix of the schedule has updated.

    This is the value actually fed to C/D in the diamond tick.  It is included
    because it is often the operationally relevant source variable, even when
    the CLI asks for the pre-tick boundary value.
    """
    prefix = tuple(v for v in schedule if ds.offB <= int(v) < ds.offC)
    if len(prefix) == 0:
        after = np.asarray(states, dtype=np.int64)
    else:
        after = ds.joint.step_map(prefix)[np.asarray(states, dtype=np.int64)]
    b_vertices = tuple(src for src, _ in ds.interface_BC[: ds.w])
    return _encode_word(_word_digits(after, ds.q, b_vertices), ds.q)


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #
def audit_winner_basin_source_charge(
    winner_path: str,
    *,
    winner_index: int = 6,
    schedule_index: int = 0,
    selection_initial_pool: InitialPool = "all",
    selection_observer_init: ObserverInit = "zero",
    run_initial_pool: Literal["all_joint", "declared"] = "all_joint",
    run_pool: InitialPool = "all",
    run_observer_init: ObserverInit = "all",
    source_mode: SourceMode = "both",
    max_steps: int = 128,
    max_exact_states: int = 250_000,
    save_assignments: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Audit one saved winner and return (summary, per-state dataframe)."""
    records = BS.load_winner_records(winner_path)
    if int(winner_index) < 0 or int(winner_index) >= len(records):
        raise IndexError(f"winner_index={winner_index} out of range for {len(records)} saved winners")
    rec = records[int(winner_index)]
    ds = rec.get("diamond")
    if ds is None:
        raise ValueError(f"winner record {winner_index} has no diamond object")
    if ds.n_joint_states > int(max_exact_states):
        raise ValueError(
            f"joint state space {ds.n_joint_states} exceeds max_exact_states={max_exact_states}; "
            "increase cap if this exact audit is intended"
        )

    # Reconstruct the exact selected alphabet S from the same selection manifold
    # used by blind/common-factor audits.
    cf, maps, infos, selection_states = AA.selected_common_factor(
        ds,
        initial_pool=selection_initial_pool,
        observer_init=selection_observer_init,
    )
    if int(schedule_index) < 0 or int(schedule_index) >= len(infos):
        raise IndexError(f"schedule_index={schedule_index} out of range for {len(infos)} schedules")
    T = np.asarray(maps[int(schedule_index)], dtype=np.int64)
    schedule = infos[int(schedule_index)].schedule

    if run_initial_pool == "all_joint":
        init_states = _all_joint_states(ds)
    elif run_initial_pool == "declared":
        init_states = _declared_initial_states(ds, run_pool, run_observer_init)
    else:
        raise ValueError("run_initial_pool must be 'all_joint' or 'declared'")

    # Exact functional-graph view: avoids per-state repeated loops and detects
    # non-fixed cycles exactly.
    fg = AA.functional_graph_decomposition(T)
    fixed_cycle = np.array([len(cyc) == 1 for cyc in fg.cycles], dtype=bool)
    cycle_id = fg.cycle_id[init_states]
    hit_fixed = fixed_cycle[cycle_id]
    fixed_state_by_cycle = np.full(len(fg.cycles), -1, dtype=np.int64)
    for cid, cyc in enumerate(fg.cycles):
        if len(cyc) == 1:
            fixed_state_by_cycle[cid] = int(cyc[0])
    fixed_states = fixed_state_by_cycle[cycle_id]

    # Read the selected S label on fixed points, using the frozen S maps.
    stable_s = np.full(len(init_states), -1, dtype=np.int64)
    valid_s = np.zeros(len(init_states), dtype=bool)
    if np.any(hit_fixed):
        z_fixed, valid_fixed = AA.labels_for_states(ds, fixed_states[hit_fixed], cf)
        stable_s[hit_fixed] = z_fixed
        valid_s[hit_fixed] = valid_fixed

    b_pre = pre_b_boundary_words(ds, init_states)
    b_post = post_b_boundary_words(ds, init_states, schedule)

    # Main requested source is pre-tick B boundary.  Also compute post-B source,
    # because it is the value actually delivered to branches in this schedule.
    def source_stats(src: np.ndarray, name: str) -> dict:
        ok = hit_fixed & valid_s & (stable_s >= 0) & (src >= 0)
        h_s = _entropy_from_counts(np.bincount(stable_s[ok], minlength=max(1, int(cf["shared_classes"])))) if np.any(ok) else math.nan
        h_src = _entropy_from_counts(np.bincount(src[ok], minlength=ds.q ** ds.w)) if np.any(ok) else math.nan
        mi = _mutual_info_discrete(src[ok], stable_s[ok]) if np.any(ok) else math.nan
        ratio = float(mi / h_s) if (not math.isnan(mi) and not math.isnan(h_s) and h_s > 1e-12) else math.nan
        ratio_src = float(mi / h_src) if (not math.isnan(mi) and not math.isnan(h_src) and h_src > 1e-12) else math.nan
        return {
            f"{name}_entropy_bits": float(h_src) if not math.isnan(h_src) else math.nan,
            "stable_s_entropy_bits": float(h_s) if not math.isnan(h_s) else math.nan,
            f"mi_{name}_to_stable_s_bits": float(mi) if not math.isnan(mi) else math.nan,
            f"mi_{name}_to_stable_s_over_Hs": ratio,
            f"mi_{name}_to_stable_s_over_Hsource": ratio_src,
        }

    stats_pre = source_stats(b_pre, "b_initial_boundary")
    stats_post = source_stats(b_post, "b_post_update_boundary")

    ok_main = hit_fixed & valid_s & (stable_s >= 0)
    counts_s = np.bincount(stable_s[ok_main], minlength=max(1, int(cf["shared_classes"]))) if np.any(ok_main) else np.zeros(max(1, int(cf.get("shared_classes", 1))), dtype=np.int64)
    stable_entropy = _entropy_from_counts(counts_s)
    stable_entropy_norm = float(stable_entropy / math.log2(int(cf["shared_classes"]))) if int(cf["shared_classes"]) > 1 else 0.0
    used_stable_classes = int(np.count_nonzero(counts_s))

    # Contingency for the requested pre-boundary variable.
    q_side = int(ds.q ** ds.w)
    n_s = int(cf["shared_classes"])
    contingency = np.zeros((q_side, max(1, n_s)), dtype=np.int64)
    ok_cont = ok_main & (b_pre >= 0) & (b_pre < q_side) & (stable_s >= 0) & (stable_s < max(1, n_s))
    np.add.at(contingency, (b_pre[ok_cont], stable_s[ok_cont]), 1)

    verdict_ratio = stats_pre["mi_b_initial_boundary_to_stable_s_over_Hs"]
    if not math.isnan(verdict_ratio) and verdict_ratio >= 0.50:
        verdict = "LIVE ATTRACTOR CHARGE: stable S strongly depends on B's initial boundary"
    elif not math.isnan(verdict_ratio) and verdict_ratio >= 0.20:
        verdict = "PARTIAL BASIN-SOURCE CORRELATION: stable S weakly/moderately depends on B's initial boundary"
    else:
        verdict = "SOURCE-DEAD BASIN LABEL: stable S is not explained by B's initial boundary"

    summary = {
        "verdict": verdict,
        "winner_index": int(winner_index),
        "schedule_index": int(schedule_index),
        "schedule_branch_order": str(infos[int(schedule_index)].branch_order),
        "schedule_pair_id": int(infos[int(schedule_index)].pair_id),
        "q": int(ds.q),
        "w": int(ds.w),
        "k_total": int(ds.k_total),
        "n_initial_states": int(len(init_states)),
        "n_selection_states": int(len(selection_states)),
        "n_attractors_total": int(len(fg.cycles)),
        "n_fixed_points_total": int(np.sum(fixed_cycle)),
        "fixed_point_hit_fraction": float(np.mean(hit_fixed)) if len(init_states) else math.nan,
        "valid_stable_s_fraction": float(np.mean(valid_s)) if len(init_states) else math.nan,
        "valid_fixed_stable_s_fraction": float(np.mean(valid_s[hit_fixed])) if np.any(hit_fixed) else math.nan,
        "exact_shared_classes": int(cf["shared_classes"]),
        "exact_residual_qcoords": float(cf["residual_qcoords"]),
        "used_stable_s_classes": used_stable_classes,
        "stable_s_class_counts": [int(x) for x in counts_s.tolist()],
        "stable_s_entropy_bits": float(stable_entropy),
        "stable_s_entropy_norm": float(stable_entropy_norm),
        "contingency_b_initial_boundary_by_stable_s": contingency.tolist(),
    }
    summary.update(stats_pre)
    summary.update(stats_post)

    # Include saved search metadata if available.
    metrics = rec.get("metrics", {}) or {}
    for key in ("run", "generation", "candidate", "blind_fitness", "exact_residual_qcoords", "target_mode", "fitness_mode"):
        if key in metrics:
            summary[f"saved_{key}"] = metrics[key]

    comp_words = _states_to_component_words(ds, init_states)
    df = pd.DataFrame({
        "initial_state": init_states.astype(np.int64),
        "B_initial_boundary": b_pre.astype(np.int64),
        "B_post_update_boundary": b_post.astype(np.int64),
        "stable_S_label": stable_s.astype(np.int64),
        "valid_stable_S": valid_s.astype(bool),
        "hit_fixed_point": hit_fixed.astype(bool),
        "fixed_point_state": fixed_states.astype(np.int64),
        "cycle_id": cycle_id.astype(np.int64),
        "distance_to_attractor": fg.distance_to_cycle[init_states].astype(np.int64),
        **comp_words,
    })
    if save_assignments:
        os.makedirs(os.path.dirname(save_assignments) or ".", exist_ok=True)
        df.to_csv(save_assignments, index=False)
    return summary, df


# --------------------------------------------------------------------------- #
# Plotting / CLI
# --------------------------------------------------------------------------- #
def plot_basin_source_charge(assignments: pd.DataFrame, summary: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(assignments) == 0:
        return
    ok = assignments["hit_fixed_point"].astype(bool) & assignments["valid_stable_S"].astype(bool)
    df = assignments[ok].copy()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    # Contingency heatmap: rows B initial boundary, columns stable S.
    mat = np.asarray(summary.get("contingency_b_initial_boundary_by_stable_s", []), dtype=float)
    if mat.size:
        ax[0].imshow(mat, aspect="auto")
        ax[0].set_title("B initial boundary vs stable S")
        ax[0].set_xlabel("stable S label")
        ax[0].set_ylabel("B initial boundary word")
    else:
        ax[0].set_title("no valid contingency")

    if len(df):
        counts = df["stable_S_label"].value_counts().sort_index()
        ax[1].bar(counts.index.astype(str), counts.values)
    ax[1].set_title("stable S distribution")
    ax[1].set_xlabel("stable S label")
    ax[1].set_ylabel("count")

    ratios = [
        summary.get("mi_b_initial_boundary_to_stable_s_over_Hs", math.nan),
        summary.get("mi_b_post_update_boundary_to_stable_s_over_Hs", math.nan),
    ]
    ax[2].bar(["pre-B", "post-B"], ratios)
    ax[2].set_ylim(0, 1)
    ax[2].set_title("source-to-stable-S liveness")
    ax[2].set_ylabel(r"$I(source;S_{stable})/H(S_{stable})$")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit whether stable attractor S depends on B's initial boundary")
    p.add_argument("winner_path", help="pickle file written by blindselection --save-winners")
    p.add_argument("--winner-index", type=int, default=6)
    p.add_argument("--schedule-index", type=int, default=0)
    p.add_argument("--selection-initial-pool", choices=["all", "rec"], default="all")
    p.add_argument("--selection-observer-init", choices=["zero", "all"], default="zero")
    p.add_argument("--run-initial-pool", choices=["all_joint", "declared"], default="all_joint")
    p.add_argument("--run-pool", choices=["all", "rec"], default="all")
    p.add_argument("--run-observer-init", choices=["zero", "all"], default="all")
    p.add_argument("--max-steps", type=int, default=128, help="reserved for compatibility; functional graph is exact")
    p.add_argument("--max-exact-states", type=int, default=250000)
    p.add_argument("--out", default="example_results/basin_source_charge.csv", help="per-initial-state assignments CSV")
    p.add_argument("--summary", default=None, help="summary JSON; defaults to OUT stem + _summary.json")
    p.add_argument("--plot", default=None)
    args = p.parse_args(argv)

    summary, df = audit_winner_basin_source_charge(
        args.winner_path,
        winner_index=args.winner_index,
        schedule_index=args.schedule_index,
        selection_initial_pool=args.selection_initial_pool,
        selection_observer_init=args.selection_observer_init,
        run_initial_pool=args.run_initial_pool,
        run_pool=args.run_pool,
        run_observer_init=args.run_observer_init,
        max_steps=args.max_steps,
        max_exact_states=args.max_exact_states,
        save_assignments=args.out,
    )
    print(json.dumps(summary, indent=2, default=float))

    summary_path = args.summary or (args.out.rsplit(".", 1)[0] + "_summary.json" if args.out else "example_results/basin_source_charge_summary.json")
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
    if args.plot:
        plot_basin_source_charge(df, summary, args.plot)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
