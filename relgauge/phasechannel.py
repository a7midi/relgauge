"""
phasechannel.py -- finite observer channels for another observer's schedule phase.

This is the next experiment after finiteobserver.py.

finiteobserver.py measured how much of B's *bulk state* a finite SCC observer A
can retain through a feed-forward interface B -> A.  phasechannel.py asks a
narrower question suggested by the gauge-dimension result:

    Does A retain information about B's one schedule-gauge phase?

The experiment keeps the closure principle: A is not an infinite history tape.
A is an actual finite SCC with q**kA memory states, evolving under its own
rules while it receives boundary input from B.

We test three finite phase variables:

  cut          phi = first-updated vertex of B in the first tick.  This is the
               structural temporal cut / schedule-label phase.

  erased_value phi = old value of a fixed B vertex v_phi, with the first B
               update forced to be v_phi.  For cycle-like in-place semantics,
               this probes the q-ary coordinate that is overwritten before it
               can propagate.  It is the closest finite proxy for the
               one-coordinate gauge deficit.

  cut_value    phi = (first-updated B vertex, old value at that vertex).  This
               is a combined schedule-label + q-ary erased-coordinate probe.

For each phi, nuisance variables are averaged/marginalized: B's recurrent
initial state consistent with phi, A's initial state, and admissible schedules.
The output is only A's final internal state after T ticks.

Main output columns:

  phase_mi_bits_uniform      I(Phi ; M_A(T)) for a uniform prior over available
                             phase labels.
  phase_capacity_bits        Shannon capacity of the finite phase->memory
                             channel.
  phase_zero_error_bits      one-shot zero-error distinguishability of phase
                             labels from A's final memory.
  bulk_capacity_bits         optional comparison: finiteobserver capacity for
                             the full B bulk state in the same instance.

Interpretation:
  If bulk capacity is bottlenecked but phase information survives, then the
  theory's surviving object is not bulk-state holography but inter-observer
  phase consistency.  If phase information also decays, schedule phase is not
  boundary-physical for this observer ensemble/regime.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import core as C
from . import finiteobserver as F

PhaseMode = Literal["cut", "erased_value", "cut_value"]
PhaseProtocol = Literal["first", "same"]


@dataclass(frozen=True)
class PhaseChannel:
    labels: tuple
    P: np.ndarray                    # P[A_final | phase label]
    possible_outputs: dict           # label -> frozenset[A_final]
    mode: str
    protocol: str
    phase_vertex: int | None


# --------------------------------------------------------------------------- #
# schedule groups / phase labels
# --------------------------------------------------------------------------- #
def _b_first_vertex(co: F.CoupledObserver, schedule: tuple[int, ...]) -> int:
    """First B vertex updated in an admissible B-before-A schedule."""
    for v in schedule:
        if 0 <= v < co.kB:
            return int(v)
    raise ValueError("schedule contains no B vertex")


def schedule_groups_by_cut(co: F.CoupledObserver) -> tuple[list[tuple[int, ...]], dict[int, np.ndarray]]:
    """Return all admissible schedules and indices grouped by B's first vertex."""
    schedules = F.admissible_two_scc_schedules(co)
    groups: dict[int, list[int]] = {v: [] for v in range(co.kB)}
    for i, sched in enumerate(schedules):
        groups[_b_first_vertex(co, sched)].append(i)
    return schedules, {k: np.asarray(v, dtype=np.int64) for k, v in groups.items() if v}


def _digit(indices: np.ndarray, q: int, vertex: int) -> np.ndarray:
    return (indices // (q ** vertex)) % q


def _available_labels(
    co: F.CoupledObserver,
    b_rec: np.ndarray,
    mode: PhaseMode,
    phase_vertex: int | None,
) -> tuple:
    """Phase labels that actually occur on B's recurrent set."""
    if mode == "cut":
        return tuple(range(co.kB))
    if mode == "erased_value":
        if phase_vertex is None:
            phase_vertex = 0
        vals = sorted(set(_digit(b_rec, co.q, int(phase_vertex)).astype(int).tolist()))
        return tuple(vals)
    if mode == "cut_value":
        labels = []
        for cut in range(co.kB):
            vals = sorted(set(_digit(b_rec, co.q, cut).astype(int).tolist()))
            labels.extend((cut, val) for val in vals)
        return tuple(labels)
    raise ValueError(f"unknown phase mode: {mode!r}")


def _initial_b_states_for_label(
    co: F.CoupledObserver,
    b_rec: np.ndarray,
    label,
    mode: PhaseMode,
    phase_vertex: int | None,
) -> np.ndarray:
    """B recurrent initial states compatible with a phase label."""
    b_rec = np.asarray(b_rec, dtype=np.int64)
    if mode == "cut":
        return b_rec
    if mode == "erased_value":
        if phase_vertex is None:
            phase_vertex = 0
        return b_rec[_digit(b_rec, co.q, int(phase_vertex)) == int(label)]
    if mode == "cut_value":
        cut, value = label
        return b_rec[_digit(b_rec, co.q, int(cut)) == int(value)]
    raise ValueError(f"unknown phase mode: {mode!r}")


def _first_tick_cut_for_label(
    label,
    mode: PhaseMode,
    phase_vertex: int | None,
) -> int:
    """Which first-B-update schedule group realizes a label on the first tick."""
    if mode == "cut":
        return int(label)
    if mode == "erased_value":
        return 0 if phase_vertex is None else int(phase_vertex)
    if mode == "cut_value":
        return int(label[0])
    raise ValueError(f"unknown phase mode: {mode!r}")


# --------------------------------------------------------------------------- #
# stochastic and set-valued propagation
# --------------------------------------------------------------------------- #
def _advance_distribution(dist: np.ndarray, step_maps_subset: np.ndarray) -> np.ndarray:
    """Uniformly apply one of step_maps_subset to a dense state distribution."""
    nz = np.flatnonzero(dist > 0)
    if len(nz) == 0:
        return np.zeros_like(dist)
    n_sched = step_maps_subset.shape[0]
    dest = step_maps_subset[:, nz].reshape(-1)
    weights = np.broadcast_to(dist[nz], (n_sched, len(nz))).reshape(-1) / float(n_sched)
    out = np.bincount(dest, weights=weights, minlength=len(dist)).astype(float)
    total = out.sum()
    if total > 0:
        out /= total
    return out


def _advance_set(states: np.ndarray, step_maps_subset: np.ndarray) -> np.ndarray:
    """Set-valued nondeterministic one-tick propagation."""
    if len(states) == 0:
        return states
    return np.unique(step_maps_subset[:, states].reshape(-1)).astype(np.int64)


def _initial_joint_distribution(
    co: F.CoupledObserver,
    b_states: np.ndarray,
    observer_init: F.InitMode,
) -> np.ndarray:
    a0 = F._a_init_indices(co, observer_init)  # intentionally reuses package helper
    joints = F._joint_index(
        co,
        np.repeat(np.asarray(b_states, dtype=np.int64), len(a0)),
        np.tile(a0, len(b_states)),
    )
    dist = np.zeros(co.n_joint_states, dtype=float)
    if len(joints) > 0:
        vals, counts = np.unique(joints, return_counts=True)
        dist[vals] = counts.astype(float) / float(counts.sum())
    return dist


def _initial_joint_set(
    co: F.CoupledObserver,
    b_states: np.ndarray,
    observer_init: F.InitMode,
) -> np.ndarray:
    a0 = F._a_init_indices(co, observer_init)
    joints = F._joint_index(
        co,
        np.repeat(np.asarray(b_states, dtype=np.int64), len(a0)),
        np.tile(a0, len(b_states)),
    )
    return np.unique(joints).astype(np.int64)


def _marginal_A(co: F.CoupledObserver, dist: np.ndarray) -> np.ndarray:
    nz = np.flatnonzero(dist > 0)
    out = np.zeros(co.nA_states, dtype=float)
    if len(nz):
        a = F._a_part(co, nz)
        out += np.bincount(a, weights=dist[nz], minlength=co.nA_states)
    total = out.sum()
    if total > 0:
        out /= total
    return out


def phase_channel_matrix(
    co: F.CoupledObserver,
    horizon: int,
    mode: PhaseMode = "cut",
    protocol: PhaseProtocol = "first",
    observer_init: F.InitMode = "zero",
    phase_vertex: int | None = None,
    step_maps: np.ndarray | None = None,
    b_rec: np.ndarray | None = None,
) -> PhaseChannel:
    """Exact finite channel P(A_T | Phi) for an observer phase variable.

    protocol='first': Phi fixes only the first tick's B schedule cut; later
        ticks use all admissible schedules as gauge/noise.
    protocol='same': Phi fixes the same B schedule cut at every tick.  This is
        a persistence/control variant, useful as an upper bound on phase
        retention.
    """
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    if protocol not in ("first", "same"):
        raise ValueError("protocol must be 'first' or 'same'")
    if step_maps is None:
        step_maps = F.step_maps_coupled(co)
    if b_rec is None:
        b_rec = F.observed_recurrent_states(co)
    _, cut_groups = schedule_groups_by_cut(co)
    all_sched = np.arange(step_maps.shape[0], dtype=np.int64)

    labels = _available_labels(co, b_rec, mode, phase_vertex)
    rows: list[np.ndarray] = []
    possible: dict = {}
    kept_labels: list = []

    for label in labels:
        cut = _first_tick_cut_for_label(label, mode, phase_vertex)
        group_idx = cut_groups.get(cut)
        if group_idx is None or len(group_idx) == 0:
            continue
        b_states = _initial_b_states_for_label(co, b_rec, label, mode, phase_vertex)
        if len(b_states) == 0:
            continue

        # Stochastic channel row.
        dist = _initial_joint_distribution(co, b_states, observer_init)
        for t in range(horizon):
            idx = group_idx if (protocol == "same" or t == 0) else all_sched
            dist = _advance_distribution(dist, step_maps[idx])
        rows.append(_marginal_A(co, dist))

        # Zero-error possible-output row.
        states = _initial_joint_set(co, b_states, observer_init)
        for t in range(horizon):
            idx = group_idx if (protocol == "same" or t == 0) else all_sched
            states = _advance_set(states, step_maps[idx])
        a_final = np.unique(F._a_part(co, states)).astype(np.int64)
        possible[label] = frozenset(int(x) for x in a_final.tolist())
        kept_labels.append(label)

    if rows:
        P = np.vstack(rows)
        # Numerical cleanup.
        rs = P.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        P = P / rs
    else:
        P = np.zeros((0, co.nA_states), dtype=float)
    return PhaseChannel(
        labels=tuple(kept_labels),
        P=P,
        possible_outputs=possible,
        mode=mode,
        protocol=protocol,
        phase_vertex=phase_vertex,
    )


# --------------------------------------------------------------------------- #
# information measures
# --------------------------------------------------------------------------- #
def mutual_information_uniform(P: np.ndarray) -> float:
    """I(X;Y) in bits for uniform X and channel rows P(Y|X)."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] == 0:
        return float("nan")
    P = P[:, P.sum(axis=0) > 0]
    if P.shape[1] == 0:
        return 0.0
    P = P / P.sum(axis=1, keepdims=True)
    py = P.mean(axis=0)
    eps = 1e-300
    mask = P > 0
    log_ratio = np.zeros_like(P)
    log_ratio[mask] = np.log2(P[mask]) - np.log2(np.broadcast_to(py + eps, P.shape)[mask])
    return max(0.0, float(np.sum(P * log_ratio) / P.shape[0]))


def phase_channel_measure(
    co: F.CoupledObserver,
    horizon: int,
    mode: PhaseMode = "cut",
    protocol: PhaseProtocol = "first",
    observer_init: F.InitMode = "zero",
    phase_vertex: int | None = None,
    max_exact_zero_error: int = 80,
    do_bulk: bool = True,
) -> dict:
    """Measure how much of B's phase survives in A's finite memory."""
    sm = F.step_maps_coupled(co)
    b_rec = F.observed_recurrent_states(co)
    ch = phase_channel_matrix(
        co,
        horizon=horizon,
        mode=mode,
        protocol=protocol,
        observer_init=observer_init,
        phase_vertex=phase_vertex,
        step_maps=sm,
        b_rec=b_rec,
    )
    n_labels = len(ch.labels)
    phase_entropy = float(np.log2(n_labels)) if n_labels > 0 else float("nan")
    phase_mi = mutual_information_uniform(ch.P)
    phase_cap = F.blahut_arimoto_capacity(ch.P) if n_labels > 0 else float("nan")
    ze_size, ze_exact = F.zero_error_code_size(ch.possible_outputs, max_exact=max_exact_zero_error)
    ze_bits = float(np.log2(max(1, ze_size)))

    memory_bits = co.memory_bits
    cut_bits = horizon * co.boundary_width * float(np.log2(co.q))
    physical_bound = min(phase_entropy, memory_bits, cut_bits) if np.isfinite(phase_entropy) else float("nan")

    bulk_cap = float("nan")
    bulk_ze = float("nan")
    bulk_log_rec = float(np.log2(len(b_rec))) if len(b_rec) else float("nan")
    if do_bulk:
        bm = F.finite_observer_measure(co, horizon=horizon, observer_init=observer_init, do_shannon=True)
        bulk_cap = float(bm["shannon_capacity_bits"])
        bulk_ze = float(bm["zero_error_bits"])
        bulk_log_rec = float(bm["logRecB_bits"])

    return dict(
        kB=co.kB,
        kA=co.kA,
        q=co.q,
        w=co.boundary_width,
        horizon=int(horizon),
        phase_mode=mode,
        phase_protocol=protocol,
        phase_vertex=(-1 if phase_vertex is None else int(phase_vertex)),
        observer_init=observer_init,
        n_phase_labels=int(n_labels),
        phase_labels=str(ch.labels),
        phase_entropy_bits=phase_entropy,
        phase_mi_bits_uniform=float(phase_mi),
        phase_capacity_bits=float(phase_cap),
        phase_zero_error_code_size=int(ze_size),
        phase_zero_error_exact=bool(ze_exact),
        phase_zero_error_bits=ze_bits,
        memory_bits=float(memory_bits),
        cut_bits=float(cut_bits),
        physical_bound_bits=float(physical_bound),
        phase_mi_over_entropy=float(phase_mi / phase_entropy) if phase_entropy > 0 else float("nan"),
        phase_capacity_over_entropy=float(phase_cap / phase_entropy) if phase_entropy > 0 else float("nan"),
        phase_capacity_per_boundary_symbol=float(phase_cap / (co.boundary_width * np.log2(co.q)))
        if co.boundary_width > 0 and np.isfinite(phase_cap)
        else float("nan"),
        bulk_capacity_bits=bulk_cap,
        bulk_zero_error_bits=bulk_ze,
        logRecB_bits=bulk_log_rec,
        phase_capacity_over_bulk_capacity=float(phase_cap / bulk_cap)
        if bulk_cap > 1e-12 and np.isfinite(phase_cap)
        else float("nan"),
        phase_capacity_over_volume=float(phase_cap / bulk_log_rec)
        if bulk_log_rec > 0 and np.isfinite(phase_cap)
        else float("nan"),
        n_rec_B=int(len(b_rec)),
        n_A_states=int(co.nA_states),
        n_joint_states=int(co.n_joint_states),
        n_schedules=int(sm.shape[0]),
        nontrivial_sccs=int(F.nontrivial_scc_count(co.joint)),
        interface=str(co.interface),
    )


# --------------------------------------------------------------------------- #
# sweeps / analysis / plotting
# --------------------------------------------------------------------------- #
def _parse_modes(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def run_phase_channel_sweep(
    ks_B: Iterable[int] = (2, 3, 4),
    ks_A: Iterable[int] = (3,),
    ws: Iterable[int] = (1, 2),
    horizons: Iterable[int] = (1, 3, 6),
    q: int = 4,
    n_instances: int = 20,
    phase_modes: Iterable[PhaseMode] = ("cut", "erased_value"),
    protocols: Iterable[PhaseProtocol] = ("first",),
    observer_init: F.InitMode = "zero",
    phase_vertex: int = 0,
    extra_B: int | None = None,
    extra_A: int | None = None,
    base_seed: int = 0,
    max_joint_states: int = 4 ** 7,
    do_bulk: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for kB in ks_B:
        for kA in ks_A:
            if q ** (kB + kA) > max_joint_states:
                if verbose:
                    print(f"skip kB={kB}, kA={kA}: q^(kB+kA) too large", flush=True)
                continue
            for w in ws:
                if w > min(kB, kA):
                    continue
                for inst in range(n_instances):
                    seed = (base_seed * 130363 + kB * 9176 + kA * 3571 + w * 809 + inst) % 2**32
                    rng = np.random.default_rng(seed)
                    co = F.make_coupled_observer(kB, kA, q, w, rng, extra_B=extra_B, extra_A=extra_A)
                    for horizon in horizons:
                        for mode in phase_modes:
                            for protocol in protocols:
                                m = phase_channel_measure(
                                    co,
                                    horizon=int(horizon),
                                    mode=mode,  # type: ignore[arg-type]
                                    protocol=protocol,  # type: ignore[arg-type]
                                    observer_init=observer_init,
                                    phase_vertex=phase_vertex,
                                    do_bulk=do_bulk,
                                )
                                m.update(seed=int(seed), instance=int(inst), extra_B=co.meta["extra_B"], extra_A=co.meta["extra_A"])
                                rows.append(m)
                if verbose:
                    print(f"phase-channel kB={kB}, kA={kA}, w={w} done", flush=True)
    return pd.DataFrame(rows)


def analyze_phase_channel(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    eps = 1e-8
    phase_bound_viol = int((df["phase_capacity_bits"] > df["physical_bound_bits"] + 1e-6).sum())
    phase_mem_viol = int((df["phase_capacity_bits"] > df["memory_bits"] + eps).sum())
    phase_entropy_viol = int((df["phase_capacity_bits"] > df["phase_entropy_bits"] + eps).sum())
    ze_entropy_viol = int((df["phase_zero_error_bits"] > df["phase_entropy_bits"] + eps).sum())

    group_cols = ["phase_mode", "phase_protocol", "kA", "w", "horizon"]
    scaling = []
    for keys, sub in df.groupby(group_cols):
        by_k = sub.groupby("kB").agg(
            phase_capacity=("phase_capacity_bits", "mean"),
            phase_mi=("phase_mi_bits_uniform", "mean"),
            phase_zero_error=("phase_zero_error_bits", "mean"),
            phase_entropy=("phase_entropy_bits", "mean"),
            bulk_capacity=("bulk_capacity_bits", "mean"),
            volume=("logRecB_bits", "mean"),
        )
        if len(by_k) >= 2:
            pc_growth = float(by_k.phase_capacity.iloc[-1] - by_k.phase_capacity.iloc[0])
            vol_growth = float(by_k.volume.iloc[-1] - by_k.volume.iloc[0])
            pc_over_vol = pc_growth / vol_growth if abs(vol_growth) > eps else float("nan")
        else:
            pc_over_vol = float("nan")
        mode, protocol, kA, w, horizon = keys
        scaling.append(
            dict(
                phase_mode=str(mode),
                phase_protocol=str(protocol),
                kA=int(kA),
                w=int(w),
                horizon=int(horizon),
                phase_capacity_by_kB={int(k): float(v) for k, v in by_k.phase_capacity.items()},
                phase_mi_by_kB={int(k): float(v) for k, v in by_k.phase_mi.items()},
                phase_zero_error_by_kB={int(k): float(v) for k, v in by_k.phase_zero_error.items()},
                phase_entropy_by_kB={int(k): float(v) for k, v in by_k.phase_entropy.items()},
                bulk_capacity_by_kB={int(k): float(v) for k, v in by_k.bulk_capacity.items()},
                logRec_by_kB={int(k): float(v) for k, v in by_k.volume.items()},
                phase_capacity_growth_over_volume_growth=float(pc_over_vol),
            )
        )

    avg_phase_retention = float(df["phase_mi_over_entropy"].replace([np.inf, -np.inf], np.nan).mean())
    avg_phase_cap_retention = float(df["phase_capacity_over_entropy"].replace([np.inf, -np.inf], np.nan).mean())
    avg_phase_over_volume = float(df["phase_capacity_over_volume"].replace([np.inf, -np.inf], np.nan).mean())
    avg_phase_over_bulk = float(df["phase_capacity_over_bulk_capacity"].replace([np.inf, -np.inf], np.nan).mean())

    cut_df = df[df.phase_mode == "cut"] if "phase_mode" in df else pd.DataFrame()
    erased_df = df[df.phase_mode == "erased_value"] if "phase_mode" in df else pd.DataFrame()
    cut_ret = float(cut_df["phase_mi_over_entropy"].replace([np.inf, -np.inf], np.nan).mean()) if len(cut_df) else float("nan")
    erased_ret = float(erased_df["phase_mi_over_entropy"].replace([np.inf, -np.inf], np.nan).mean()) if len(erased_df) else float("nan")

    if phase_bound_viol or phase_mem_viol or phase_entropy_viol or ze_entropy_viol:
        verdict = "IMPLEMENTATION WARNING: a phase information bound was violated"
    elif np.isfinite(avg_phase_cap_retention) and avg_phase_cap_retention > 0.35 and avg_phase_over_volume < 0.35:
        verdict = "PHASE-SELECTIVE CHANNEL: phase survives better than bulk volume"
    elif np.isfinite(avg_phase_cap_retention) and avg_phase_cap_retention < 0.10:
        verdict = "PHASE ALSO HIDDEN/FORGOTTEN in this finite-observer regime"
    else:
        verdict = "MIXED / PARTIAL PHASE RETENTION: inspect mode, horizon, and width"

    return dict(
        verdict=verdict,
        phase_bound_violations=phase_bound_viol,
        phase_memory_bound_violations=phase_mem_viol,
        phase_entropy_bound_violations=phase_entropy_viol,
        zero_error_phase_entropy_bound_violations=ze_entropy_viol,
        mean_phase_mi_over_entropy=avg_phase_retention,
        mean_phase_capacity_over_entropy=avg_phase_cap_retention,
        mean_phase_capacity_over_volume=avg_phase_over_volume,
        mean_phase_capacity_over_bulk_capacity=avg_phase_over_bulk,
        mean_cut_phase_mi_over_entropy=cut_ret,
        mean_erased_value_phase_mi_over_entropy=erased_ret,
        scaling=scaling,
    )


def plot_phase_channel(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    fig, ax = plt.subplots(figsize=(8, 5))
    # Keep plot compact: show phase capacity vs kB for each mode at the largest horizon present.
    for (mode, w, horizon), sub in df.groupby(["phase_mode", "w", "horizon"]):
        g = sub.groupby("kB")["phase_capacity_bits"].mean().dropna()
        label = f"{mode}, w={w}, T={horizon}"
        ax.plot(g.index, g.values, "o-", label=label)
    ax.set_xlabel("observed observer size kB")
    ax.set_ylabel("phase capacity (bits)")
    ax.set_title("Finite observer phase-channel capacity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Finite observer phase-channel experiment")
    parser.add_argument("q", nargs="?", type=int, default=4)
    parser.add_argument("--ks-B", default="2,3,4", help="comma list, observed SCC sizes")
    parser.add_argument("--ks-A", default="3", help="comma list, observer SCC sizes")
    parser.add_argument("--ws", default="1,2", help="comma list, interface widths")
    parser.add_argument("--horizons", default="1,3,6", help="comma list of tick horizons")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--phase-modes", default="cut,erased_value", help="cut, erased_value, cut_value")
    parser.add_argument("--protocols", default="first", help="first, same")
    parser.add_argument("--phase-vertex", type=int, default=0)
    parser.add_argument("--observer-init", choices=["zero", "all"], default="zero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-joint-states", type=int, default=4 ** 7)
    parser.add_argument("--no-bulk", action="store_true", help="skip full-bulk finiteobserver comparison")
    parser.add_argument("--out", default="example_results/phase_channel.csv")
    parser.add_argument("--plot", default="example_results/fig_phase_channel.png")
    args = parser.parse_args(argv)

    modes = _parse_modes(args.phase_modes)
    protocols = _parse_modes(args.protocols)
    allowed_modes = {"cut", "erased_value", "cut_value"}
    allowed_protocols = {"first", "same"}
    unknown_modes = set(modes) - allowed_modes
    unknown_protocols = set(protocols) - allowed_protocols
    if unknown_modes:
        raise SystemExit(f"unknown phase mode(s): {sorted(unknown_modes)}")
    if unknown_protocols:
        raise SystemExit(f"unknown protocol(s): {sorted(unknown_protocols)}")

    df = run_phase_channel_sweep(
        ks_B=_parse_ints(args.ks_B),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        horizons=_parse_ints(args.horizons),
        q=args.q,
        n_instances=args.instances,
        phase_modes=modes,  # type: ignore[arg-type]
        protocols=protocols,  # type: ignore[arg-type]
        observer_init=args.observer_init,
        phase_vertex=args.phase_vertex,
        base_seed=args.seed,
        max_joint_states=args.max_joint_states,
        do_bulk=not args.no_bulk,
        verbose=True,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    res = analyze_phase_channel(df)
    summary_path = os.path.join(os.path.dirname(args.out) or ".", "phase_channel_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    try:
        plot_phase_channel(df, args.plot)
    except Exception as exc:
        print(f"plot skipped: {exc}")
    print(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
