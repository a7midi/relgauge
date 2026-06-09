"""
attractoraudit.py -- recurrent-orbit audit for selected interface alphabets.

This module is a Stage-1 dynamical audit for blind-selection winners.  The
exact equalizer/common-factor tools show whether a diamond has a nontrivial
shared interface alphabet S at a convergence event.  This audit asks the next
question:

    What happens to that same selected alphabet S on the recurrent dynamics?

For each saved winner, the module:
  1. Reconstructs the exact common factor S from the one-tick diamond data.
  2. Freezes the left/right maps into that selected alphabet.
  3. Chooses one or more deterministic admissible schedules.
  4. Computes the functional-graph attractors of the full joint dynamics.
  5. Reads S on the recurrent cycles and classifies it as:
       * constant on attractors  -> basin/conservation-like label;
       * periodic on attractors  -> phase-cycling label;
       * invalid/mixing          -> formal instantaneous equalizer only.

Winner files are local pickle artifacts.  Do not load winner files from
untrusted sources.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import scdcdiamond as S
from . import blindselection as BS
from . import commonfactoraudit as CFA

InitialPool = S.InitialPool
ObserverInit = S.ObserverInit
ScheduleMode = Literal["canonical", "all"]


# --------------------------------------------------------------------------- #
# Basic information helpers
# --------------------------------------------------------------------------- #
def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


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


def _label_min_period(seq: np.ndarray) -> int:
    """Minimal period of a finite cyclic sequence."""
    seq = np.asarray(seq, dtype=np.int64)
    n = int(len(seq))
    if n <= 1:
        return n
    for p in range(1, n + 1):
        if n % p == 0 and np.all(seq == np.resize(seq[:p], n)):
            return int(p)
    return n


# --------------------------------------------------------------------------- #
# Exact common-factor reconstruction and state labeling
# --------------------------------------------------------------------------- #
def selected_common_factor(
    ds: S.SCDCDiamond,
    *,
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
) -> tuple[dict, np.ndarray, list[S.ScheduleInfo], np.ndarray]:
    """Reconstruct the exact selected alphabet S for a saved diamond."""
    maps, infos = S.step_maps(ds)
    states = S._initial_states(ds, initial_pool, observer_init)
    if len(states) == 0:
        raise ValueError("candidate has no initial states under requested pool/init")
    finals = maps[:, states].reshape(-1)
    left, right = CFA.final_panel_words(ds, finals)
    cf = CFA.exact_common_factor(left, right, ds.q, ds.w)
    return cf, maps, infos, states


def labels_for_states(ds: S.SCDCDiamond, states: np.ndarray, cf: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read current A-panel state through an already selected alphabet S."""
    left, right = CFA.final_panel_words(ds, np.asarray(states, dtype=np.int64))
    return CFA.labels_in_existing_common_factor(left, right, cf)


# --------------------------------------------------------------------------- #
# Functional graph decomposition
# --------------------------------------------------------------------------- #
@dataclass
class FunctionalGraph:
    cycle_id: np.ndarray
    distance_to_cycle: np.ndarray
    cycles: list[np.ndarray]
    basin_sizes: np.ndarray


def functional_graph_decomposition(transition: np.ndarray) -> FunctionalGraph:
    """Exact cycle/basin decomposition for a deterministic finite map."""
    T = np.asarray(transition, dtype=np.int64)
    n = int(len(T))
    cycle_id = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, -1, dtype=np.int64)
    cycles: list[np.ndarray] = []

    for start in range(n):
        if cycle_id[start] >= 0:
            continue
        path: list[int] = []
        pos: dict[int, int] = {}
        x = int(start)
        while cycle_id[x] < 0 and x not in pos:
            pos[x] = len(path)
            path.append(x)
            x = int(T[x])
        if x in pos:
            j = pos[x]
            cyc = np.array(path[j:], dtype=np.int64)
            cid = len(cycles)
            cycles.append(cyc)
            for node in cyc:
                cycle_id[int(node)] = cid
                dist[int(node)] = 0
            prefix = path[:j]
        else:
            cid = int(cycle_id[x])
            prefix = path
        for node in reversed(prefix):
            cycle_id[int(node)] = cid
            dist[int(node)] = int(dist[int(T[int(node)])]) + 1

    basin_sizes = np.bincount(cycle_id, minlength=len(cycles)).astype(np.int64)
    return FunctionalGraph(cycle_id=cycle_id, distance_to_cycle=dist, cycles=cycles, basin_sizes=basin_sizes)


# --------------------------------------------------------------------------- #
# One schedule / one winner audit
# --------------------------------------------------------------------------- #
def audit_winner_schedule(
    ds: S.SCDCDiamond,
    *,
    cf: dict | None = None,
    maps: np.ndarray | None = None,
    infos: list[S.ScheduleInfo] | None = None,
    initial_states: np.ndarray | None = None,
    candidate_index: int = 0,
    schedule_index: int = 0,
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    max_exact_states: int = 250_000,
) -> dict:
    """Analyze selected S on the attractors of one deterministic schedule."""
    if cf is None or maps is None or infos is None or initial_states is None:
        cf, maps, infos, initial_states = selected_common_factor(ds, initial_pool=initial_pool, observer_init=observer_init)
    if int(schedule_index) < 0 or int(schedule_index) >= len(infos):
        raise IndexError("schedule_index out of range")
    n = int(ds.n_joint_states)
    if n > int(max_exact_states):
        raise ValueError(
            f"joint state space {n} exceeds max_exact_states={max_exact_states}; "
            "increase the cap or audit a smaller winner"
        )

    T = np.asarray(maps[int(schedule_index)], dtype=np.int64)
    fg = functional_graph_decomposition(T)
    all_states = np.arange(n, dtype=np.int64)
    z_state, valid_state = labels_for_states(ds, all_states, cf)

    n_classes = int(cf.get("shared_classes", 0))
    cycle_periods: list[int] = []
    label_periods: list[int] = []
    valid_fracs: list[float] = []
    constant_flags: list[bool] = []
    phase_flags: list[bool] = []
    stable_labels = np.full(len(fg.cycles), -1, dtype=np.int64)
    transition_pairs: list[tuple[int, int]] = []

    for cid, cyc in enumerate(fg.cycles):
        labels = z_state[cyc]
        valid = valid_state[cyc]
        period = int(len(cyc))
        cycle_periods.append(period)
        valid_frac = float(np.mean(valid)) if len(valid) else math.nan
        valid_fracs.append(valid_frac)
        if len(cyc) and bool(np.all(valid)):
            lp = _label_min_period(labels)
            label_periods.append(lp)
            is_const = (lp == 1)
            constant_flags.append(bool(is_const))
            phase_flags.append(bool(lp > 1))
            if is_const:
                stable_labels[cid] = int(labels[0])
            # label transition statistics around the cycle
            nxt = np.roll(labels, -1)
            transition_pairs.extend((int(a), int(b)) for a, b in zip(labels, nxt))
        else:
            label_periods.append(0)
            constant_flags.append(False)
            phase_flags.append(False)

    basin_sizes = fg.basin_sizes.astype(float)
    n_states = float(len(T))
    constant_basin_fraction = float(basin_sizes[np.asarray(constant_flags, dtype=bool)].sum() / n_states) if n_states else math.nan
    phase_basin_fraction = float(basin_sizes[np.asarray(phase_flags, dtype=bool)].sum() / n_states) if n_states else math.nan
    all_valid_basin_fraction = float(sum(float(basin_sizes[i]) for i, vf in enumerate(valid_fracs) if vf == 1.0) / n_states) if n_states else math.nan

    # Stable basin-label entropy over declared initial states, not all states.
    init = np.asarray(initial_states, dtype=np.int64)
    init_cycle = fg.cycle_id[init]
    init_stable = stable_labels[init_cycle]
    ok_stable = init_stable >= 0
    if np.any(ok_stable):
        counts = np.bincount(init_stable[ok_stable], minlength=max(1, n_classes))
        stable_entropy = _entropy_from_counts(counts)
        stable_entropy_norm = float(stable_entropy / math.log2(n_classes)) if n_classes > 1 else 0.0
        n_stable_labels = int(np.count_nonzero(counts))
    else:
        stable_entropy = 0.0
        stable_entropy_norm = 0.0
        n_stable_labels = 0

    # Does the starting interface label predict the eventual stable attractor label?
    z_init = z_state[init]
    mi_init_to_stable = _mutual_info_discrete(z_init, init_stable)
    hz_init = _entropy_from_counts(np.bincount(z_init[z_init >= 0], minlength=max(1, n_classes))) if np.any(z_init >= 0) else 0.0
    mi_init_to_stable_norm = float(mi_init_to_stable / hz_init) if hz_init > 1e-12 and not math.isnan(mi_init_to_stable) else math.nan

    # Attractor transition conservation and MI for valid labels on cycles.
    if transition_pairs:
        p = np.array(transition_pairs, dtype=np.int64)
        conservation_fraction = float(np.mean(p[:, 0] == p[:, 1]))
        trans_mi = _mutual_info_discrete(p[:, 0], p[:, 1])
        h0 = _entropy_from_counts(np.bincount(p[:, 0], minlength=max(1, n_classes)))
        trans_mi_norm = float(trans_mi / h0) if h0 > 1e-12 and not math.isnan(trans_mi) else math.nan
    else:
        conservation_fraction = math.nan
        trans_mi = math.nan
        trans_mi_norm = math.nan

    schedule = infos[int(schedule_index)]
    return dict(
        candidate_index=int(candidate_index),
        schedule_index=int(schedule_index),
        schedule_branch_order=str(schedule.branch_order),
        schedule_pair_id=int(schedule.pair_id),
        q=int(ds.q),
        w=int(ds.w),
        k_total=int(ds.k_total),
        kB=int(ds.kB),
        kC=int(ds.kC),
        kD=int(ds.kD),
        kA=int(ds.kA),
        exact_shared_classes=int(n_classes),
        exact_residual_qcoords=float(cf.get("residual_qcoords", math.nan)),
        n_joint_states=int(n),
        n_initial_states=int(len(init)),
        n_attractors=int(len(fg.cycles)),
        mean_cycle_period=float(np.mean(cycle_periods)) if cycle_periods else math.nan,
        max_cycle_period=int(max(cycle_periods)) if cycle_periods else 0,
        mean_label_period=float(np.mean([p for p in label_periods if p > 0])) if any(p > 0 for p in label_periods) else math.nan,
        max_label_period=int(max(label_periods)) if label_periods else 0,
        mean_cycle_label_valid_fraction=float(np.mean(valid_fracs)) if valid_fracs else math.nan,
        all_valid_basin_fraction=float(all_valid_basin_fraction),
        constant_cycle_fraction=float(np.mean(constant_flags)) if constant_flags else math.nan,
        constant_basin_fraction=float(constant_basin_fraction),
        phase_cycle_fraction=float(np.mean(phase_flags)) if phase_flags else math.nan,
        phase_basin_fraction=float(phase_basin_fraction),
        stable_label_classes=int(n_stable_labels),
        stable_label_entropy_bits=float(stable_entropy),
        stable_label_entropy_norm=float(stable_entropy_norm),
        init_label_to_stable_label_mi_bits=float(mi_init_to_stable) if not math.isnan(mi_init_to_stable) else math.nan,
        init_label_to_stable_label_mi_norm=float(mi_init_to_stable_norm) if not math.isnan(mi_init_to_stable_norm) else math.nan,
        attractor_label_conservation_fraction=float(conservation_fraction) if not math.isnan(conservation_fraction) else math.nan,
        attractor_label_transition_mi_bits=float(trans_mi) if not math.isnan(trans_mi) else math.nan,
        attractor_label_transition_mi_norm=float(trans_mi_norm) if not math.isnan(trans_mi_norm) else math.nan,
    )


# --------------------------------------------------------------------------- #
# Bundle audit / analysis / plotting
# --------------------------------------------------------------------------- #
def _schedule_indices(mode: ScheduleMode, infos: list[S.ScheduleInfo], max_schedules: int) -> list[int]:
    if mode == "canonical":
        return [0]
    if mode == "all":
        n = len(infos) if int(max_schedules) <= 0 else min(len(infos), int(max_schedules))
        return list(range(n))
    raise ValueError("schedule_mode must be 'canonical' or 'all'")


def run_attractor_audit(
    winner_path: str,
    *,
    top_n: int = 20,
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    schedule_mode: ScheduleMode = "canonical",
    max_schedules: int = 0,
    max_exact_states: int = 250_000,
    verbose: bool = True,
) -> pd.DataFrame:
    records = BS.load_winner_records(winner_path)
    if int(top_n) > 0:
        records = records[: int(top_n)]
    rows: list[dict] = []
    for i, rec in enumerate(records):
        ds = rec.get("diamond")
        if ds is None:
            raise ValueError(f"winner record {i} has no diamond object")
        cf, maps, infos, states = selected_common_factor(ds, initial_pool=initial_pool, observer_init=observer_init)
        for si in _schedule_indices(schedule_mode, infos, max_schedules=max_schedules):
            row = audit_winner_schedule(
                ds,
                cf=cf,
                maps=maps,
                infos=infos,
                initial_states=states,
                candidate_index=i,
                schedule_index=si,
                initial_pool=initial_pool,
                observer_init=observer_init,
                max_exact_states=max_exact_states,
            )
            metrics = rec.get("metrics", {}) or {}
            for key in ("run", "generation", "candidate", "blind_fitness", "fitness_mode", "target_mode"):
                if key in metrics:
                    row[f"saved_{key}"] = metrics[key]
            rows.append(row)
        if verbose:
            last = rows[-1]
            print(
                f"attractor winner={i} exact={last['exact_residual_qcoords']:.3f} "
                f"constant_basin={last['constant_basin_fraction']:.3f} "
                f"phase_basin={last['phase_basin_fraction']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def analyze_attractor_audit(
    df: pd.DataFrame,
    *,
    constant_threshold: float = 0.75,
    entropy_threshold: float = 0.25,
    phase_threshold: float = 0.50,
    transition_mi_threshold: float = 0.50,
) -> dict:
    if len(df) == 0:
        return {"verdict": "NO DATA", "n_rows": 0}
    means = {
        "mean_exact_residual_qcoords": float(df["exact_residual_qcoords"].mean()),
        "mean_n_attractors": float(df["n_attractors"].mean()),
        "mean_cycle_period": float(df["mean_cycle_period"].mean()),
        "mean_constant_basin_fraction": float(df["constant_basin_fraction"].mean()),
        "mean_phase_basin_fraction": float(df["phase_basin_fraction"].mean()),
        "mean_stable_label_entropy_norm": float(df["stable_label_entropy_norm"].mean()),
        "mean_attractor_label_conservation_fraction": float(df["attractor_label_conservation_fraction"].mean(skipna=True)),
        "mean_attractor_label_transition_mi_norm": float(df["attractor_label_transition_mi_norm"].mean(skipna=True)),
        "mean_all_valid_basin_fraction": float(df["all_valid_basin_fraction"].mean()),
    }
    conserved = (
        (df["constant_basin_fraction"] >= float(constant_threshold))
        & (df["stable_label_entropy_norm"] >= float(entropy_threshold))
    )
    phase = (
        (df["phase_basin_fraction"] >= float(phase_threshold))
        & (df["attractor_label_transition_mi_norm"].fillna(0) >= float(transition_mi_threshold))
    )
    conserved_fraction = float(conserved.mean())
    phase_fraction = float(phase.mean())
    if conserved_fraction >= 0.5:
        verdict = "ATTRACTOR-CONSERVED S: selected labels are stable basin invariants on recurrent dynamics"
    elif phase_fraction >= 0.5:
        verdict = "PHASE-CYCLING S: selected labels persist as recurrent phase variables"
    else:
        verdict = "FORMAL / NONPERSISTENT S: selected equalizers do not become attractor invariants in this regime"
    out = dict(
        verdict=verdict,
        n_rows=int(len(df)),
        conserved_attractor_fraction=conserved_fraction,
        phase_cycling_fraction=phase_fraction,
    )
    out.update(means)
    if "candidate_index" in df.columns:
        top = df.sort_values(
            ["constant_basin_fraction", "stable_label_entropy_norm", "attractor_label_transition_mi_norm"],
            ascending=False,
        ).head(10)
        out["top_candidates"] = [
            {
                "candidate_index": int(r.candidate_index),
                "schedule_index": int(r.schedule_index),
                "exact_residual_qcoords": float(r.exact_residual_qcoords),
                "constant_basin_fraction": float(r.constant_basin_fraction),
                "phase_basin_fraction": float(r.phase_basin_fraction),
                "stable_label_entropy_norm": float(r.stable_label_entropy_norm),
                "attractor_label_conservation_fraction": float(r.attractor_label_conservation_fraction) if not pd.isna(r.attractor_label_conservation_fraction) else None,
                "attractor_label_transition_mi_norm": float(r.attractor_label_transition_mi_norm) if not pd.isna(r.attractor_label_transition_mi_norm) else None,
                "mean_cycle_period": float(r.mean_cycle_period),
                "max_cycle_period": int(r.max_cycle_period),
            }
            for r in top.itertuples(index=False)
        ]
    return out


def plot_attractor_audit(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(df) == 0:
        return
    # One point per candidate/schedule row.  For all-schedules runs, candidate
    # values are repeated with different schedule_index values.
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(df))
    ax[0].bar(x, df["exact_residual_qcoords"].to_numpy(dtype=float))
    ax[0].set_title("selected exact alphabet")
    ax[0].set_xlabel("winner/schedule row")
    ax[0].set_ylabel(r"exact residual $\log_q |S|$")

    ax[1].scatter(
        df["stable_label_entropy_norm"].to_numpy(dtype=float),
        df["constant_basin_fraction"].to_numpy(dtype=float),
        alpha=0.8,
    )
    ax[1].set_title("attractor conservation test")
    ax[1].set_xlabel(r"stable-label entropy / $\log |S|$")
    ax[1].set_ylabel("constant-basin fraction")
    ax[1].set_ylim(-0.02, 1.02)

    ax[2].scatter(
        df["attractor_label_transition_mi_norm"].to_numpy(dtype=float),
        df["phase_basin_fraction"].to_numpy(dtype=float),
        alpha=0.8,
    )
    ax[2].set_title("phase-cycling test")
    ax[2].set_xlabel(r"$I(S_t;S_{t+1})/H(S_t)$ on cycles")
    ax[2].set_ylabel("phase-basin fraction")
    ax[2].set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Attractor audit for saved blind-selection winners")
    p.add_argument("winner_path", help="pickle file written by blindselection --save-winners")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--initial-pool", choices=["all", "rec"], default="all")
    p.add_argument("--observer-init", choices=["zero", "all"], default="zero")
    p.add_argument("--schedule-mode", choices=["canonical", "all"], default="canonical")
    p.add_argument("--max-schedules", type=int, default=0, help="cap schedules in --schedule-mode all; 0 means all")
    p.add_argument("--max-exact-states", type=int, default=250000)
    p.add_argument("--constant-threshold", type=float, default=0.75)
    p.add_argument("--entropy-threshold", type=float, default=0.25)
    p.add_argument("--phase-threshold", type=float, default=0.50)
    p.add_argument("--transition-mi-threshold", type=float, default=0.50)
    p.add_argument("--out", default="example_results/attractor_audit.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    df = run_attractor_audit(
        args.winner_path,
        top_n=args.top_n,
        initial_pool=args.initial_pool,
        observer_init=args.observer_init,
        schedule_mode=args.schedule_mode,
        max_schedules=args.max_schedules,
        max_exact_states=args.max_exact_states,
        verbose=not bool(args.quiet),
    )
    summary = analyze_attractor_audit(
        df,
        constant_threshold=args.constant_threshold,
        entropy_threshold=args.entropy_threshold,
        phase_threshold=args.phase_threshold,
        transition_mi_threshold=args.transition_mi_threshold,
    )
    print(json.dumps(summary, indent=2, default=float))
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(args.out.rsplit(".", 1)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        if not args.quiet:
            print(f"wrote {args.out}")
            print(f"wrote {args.out.rsplit('.', 1)[0]}_summary.json")
    if args.plot:
        import os
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_attractor_audit(df, args.plot)
        if not args.quiet:
            print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
