"""
finiteobserver.py -- closure-respecting finite observer channels.

This module is the correction to the active/infinite-memory area-law test.

Old active resolver:
    A boundary bisimulation or full trace discriminator is an ideal external
    recorder.  It can retain the complete boundary history and therefore may
    recover volume information.  That is useful as an upper bound, but it is
    not an observer inside a closed finite system.

Finite observer resolver:
    Build two coupled SCCs, B -> A.  B is the observed system.  A is the
    observer.  A has exactly q**kA internal states and evolves every tick under
    its own SCC dynamics while receiving w feed-forward boundary inputs from B.
    The only record we allow A to keep is its final internal state after T
    ticks.  No external history tape is present.

The measured channel is
    X_B(0)  --->  M_A(T)
where schedule choices and the observer's own internal dynamics are noise.  We
report finite-memory Shannon capacity and zero-error information.  Both are
bounded by the observer's memory log2(q**kA); this is the closure principle in
executable form.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import core as C


InitMode = Literal["zero", "all"]


@dataclass(frozen=True)
class CoupledObserver:
    """A two-SCC feed-forward observer channel B -> A.

    Vertex layout in the joint RelationalSystem:
        B vertices: 0, ..., kB-1
        A vertices: kB, ..., kB+kA-1

    The structural graph has no A -> B edges, so B and A remain distinct SCCs
    in the condensation DAG.  All admissible schedules update B internally in
    an arbitrary order, then A internally in an arbitrary order.
    """

    observed: C.RelationalSystem
    joint: C.RelationalSystem
    kB: int
    kA: int
    q: int
    interface: tuple[tuple[int, int], ...]  # pairs (B_source, A_local_target)
    meta: dict

    @property
    def nA_states(self) -> int:
        return self.q ** self.kA

    @property
    def nB_states(self) -> int:
        return self.q ** self.kB

    @property
    def n_joint_states(self) -> int:
        return self.q ** (self.kB + self.kA)

    @property
    def memory_bits(self) -> float:
        return self.kA * float(np.log2(self.q))

    @property
    def boundary_width(self) -> int:
        return len(self.interface)


def _random_rule(q: int, indeg: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, q, size=q ** indeg, dtype=np.int64)


def _edge_adj(sys: C.RelationalSystem) -> dict[int, set[int]]:
    adj = {v: set() for v in range(sys.k)}
    for v in range(sys.k):
        for p in sys.preds[v]:
            adj[p].add(v)
    return adj


def nontrivial_scc_count(sys: C.RelationalSystem) -> int:
    """Number of SCCs with at least two vertices."""
    comps = C.tarjan_scc(_edge_adj(sys), range(sys.k))
    return sum(1 for comp in comps if len(comp) >= 2)


def make_coupled_observer(
    kB: int,
    kA: int,
    q: int,
    w: int,
    rng: np.random.Generator,
    extra_B: int | None = None,
    extra_A: int | None = None,
) -> CoupledObserver:
    """Construct a two-observer channel B -> A.

    Parameters
    ----------
    kB, kA:
        Sizes of observed and observer SCCs.  Use kA >= 2 for a genuine SCC
        observer; kA=1 is allowed only as a degenerate memory cell.
    q:
        Alphabet size at each vertex.
    w:
        Interface width: number of feed-forward B -> A boundary edges.  The
        implementation uses the first w B vertices feeding the first w A
        vertices; w is clipped to min(kB, kA).
    extra_B, extra_A:
        Extra internal feedback edges beyond the cycle backbone.  Default kB
        and kA respectively, i.e. a moderately dense SCC family.
    """
    if kB < 1 or kA < 1:
        raise ValueError("kB and kA must be positive")
    if q < 2:
        raise ValueError("q must be at least 2")
    w = int(max(0, min(w, kB, kA)))
    if w == 0:
        raise ValueError("w must be positive after clipping to min(kB, kA)")
    if extra_B is None:
        extra_B = kB
    if extra_A is None:
        extra_A = kA

    # B is the observed SCC.  It has no incoming edges from A in the joint graph.
    B = C.make_random_scc(kB, q, int(extra_B), rng)

    # A supplies the observer's internal SCC skeleton.  We reuse its internal
    # predecessor graph, but regenerate A's rules after adding B-boundary inputs.
    A_skel = C.make_random_scc(kA, q, int(extra_A), rng)

    pred_sets: list[set[int]] = [set(p) for p in B.preds]
    pred_sets.extend(set(kB + p for p in A_skel.preds[a]) for a in range(kA))

    interface: list[tuple[int, int]] = []
    for j in range(w):
        b_src = j
        a_tgt = j
        pred_sets[kB + a_tgt].add(b_src)
        interface.append((b_src, a_tgt))

    preds = [tuple(sorted(s)) for s in pred_sets]
    rules: list[np.ndarray] = []
    for v in range(kB + kA):
        if v < kB:
            rules.append(B.rules[v].copy())
        else:
            rules.append(_random_rule(q, len(preds[v]), rng))

    joint = C.RelationalSystem(
        kB + kA,
        q,
        preds,
        rules,
        meta={
            "ensemble": "coupled_observer",
            "kB": kB,
            "kA": kA,
            "w": w,
            "extra_B": int(extra_B),
            "extra_A": int(extra_A),
        },
    )
    return CoupledObserver(
        observed=B,
        joint=joint,
        kB=kB,
        kA=kA,
        q=q,
        interface=tuple(interface),
        meta=dict(joint.meta),
    )


def admissible_two_scc_schedules(co: CoupledObserver) -> list[tuple[int, ...]]:
    """All condensation-respecting vertex schedules for B -> A.

    B must be updated before A, but each SCC's internal order is free.
    """
    B_vertices = tuple(range(co.kB))
    A_vertices = tuple(range(co.kB, co.kB + co.kA))
    out: list[tuple[int, ...]] = []
    for sB in itertools.permutations(B_vertices):
        for sA in itertools.permutations(A_vertices):
            out.append(tuple(sB + sA))
    return out


def step_maps_coupled(co: CoupledObserver) -> np.ndarray:
    """One-tick maps for every admissible B-before-A schedule."""
    scheds = admissible_two_scc_schedules(co)
    M = co.n_joint_states
    out = np.empty((len(scheds), M), dtype=np.int64)
    for i, sched in enumerate(scheds):
        out[i] = co.joint.step_map(sched)
    return out


def observed_recurrent_states(co: CoupledObserver) -> np.ndarray:
    """Recurrent states of B under B's own schedule-lift."""
    sm_B = co.observed.all_step_maps()
    adj_B = C.orbit_adjacency_fast(sm_B)
    return C.recurrent_states(adj_B)


def _a_init_indices(co: CoupledObserver, mode: InitMode = "zero") -> np.ndarray:
    if mode == "zero":
        return np.array([0], dtype=np.int64)
    if mode == "all":
        return np.arange(co.nA_states, dtype=np.int64)
    raise ValueError(f"unknown observer init mode: {mode!r}")


def _joint_index(co: CoupledObserver, b_index: np.ndarray, a_index: np.ndarray) -> np.ndarray:
    """Combine B and A component indices into joint indices.

    Because B occupies the low-order base-q digits, joint = b + a*q**kB.
    """
    return b_index + a_index * (co.q ** co.kB)


def _a_part(co: CoupledObserver, joint_indices: np.ndarray) -> np.ndarray:
    return joint_indices // (co.q ** co.kB)


def possible_observer_memories(
    co: CoupledObserver,
    horizon: int,
    observer_init: InitMode = "zero",
    step_maps: np.ndarray | None = None,
    b_inputs: np.ndarray | None = None,
) -> dict[int, frozenset[int]]:
    """Set-valued finite observer channel x_B(0) -> possible A memories.

    Schedule choices at every tick are treated as gauge/noise.  The observer is
    finite because only A's final internal state is retained; the full boundary
    history is discarded by construction.
    """
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    if step_maps is None:
        step_maps = step_maps_coupled(co)
    if b_inputs is None:
        b_inputs = observed_recurrent_states(co)
    a0 = _a_init_indices(co, observer_init)

    out: dict[int, frozenset[int]] = {}
    qkB = co.q ** co.kB
    for b in b_inputs.tolist():
        current = _joint_index(co, np.array([b], dtype=np.int64), a0)
        current = np.unique(current)
        for _ in range(horizon):
            # Apply every admissible one-tick schedule to every currently
            # possible joint state.  This is exact nondeterministic propagation.
            current = np.unique(step_maps[:, current].reshape(-1))
        a_final = np.unique(current // qkB).astype(np.int64)
        out[int(b)] = frozenset(int(x) for x in a_final.tolist())
    return out


def stochastic_observer_channel(
    co: CoupledObserver,
    horizon: int,
    observer_init: InitMode = "zero",
    step_maps: np.ndarray | None = None,
    b_inputs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return P(A_T | B_0) under uniform schedules and uniform allowed A_0.

    This is the Shannon-capacity companion to the zero-error set-valued channel.
    It still respects closure: the output alphabet is A's finite memory state,
    not an external history tape.
    """
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    if step_maps is None:
        step_maps = step_maps_coupled(co)
    if b_inputs is None:
        b_inputs = observed_recurrent_states(co)
    b_inputs = np.asarray(b_inputs, dtype=np.int64)
    a0 = _a_init_indices(co, observer_init)
    n_sched = step_maps.shape[0]
    P = np.zeros((len(b_inputs), co.nA_states), dtype=float)

    for row, b in enumerate(b_inputs.tolist()):
        dist: dict[int, float] = {}
        p0 = 1.0 / len(a0)
        for a in a0.tolist():
            dist[int(_joint_index(co, np.array([b]), np.array([a]))[0])] = p0
        for _ in range(horizon):
            nxt: defaultdict[int, float] = defaultdict(float)
            weight = 1.0 / n_sched
            for state, prob in dist.items():
                ys = step_maps[:, state]
                for y in ys.tolist():
                    nxt[int(y)] += prob * weight
            dist = dict(nxt)
        for state, prob in dist.items():
            P[row, int(_a_part(co, np.array([state]))[0])] += prob

    # Numerical cleanup.
    rowsum = P.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    P = P / rowsum
    return b_inputs, P


def blahut_arimoto_capacity(
    P: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 2000,
) -> float:
    """Shannon capacity in bits for a finite discrete memoryless channel.

    P has shape (n_inputs, n_outputs), rows P(y|x).  The implementation uses a
    standard Blahut-Arimoto fixed-point iteration and returns max_p I(X;Y).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] == 0:
        return float("nan")
    # Drop impossible output columns to avoid log(0/0) noise.
    colsum = P.sum(axis=0)
    P = P[:, colsum > 0]
    if P.shape[1] == 0:
        return 0.0
    P = P / P.sum(axis=1, keepdims=True)
    n_inputs = P.shape[0]
    p = np.full(n_inputs, 1.0 / n_inputs)
    eps = 1e-300
    for _ in range(max_iter):
        qy = p @ P
        log_ratio = np.zeros_like(P)
        mask = P > 0
        log_qy = np.broadcast_to(np.log(qy + eps), P.shape)
        log_ratio[mask] = np.log(P[mask]) - log_qy[mask]
        D = np.sum(P * log_ratio, axis=1)  # nats
        weights = p * np.exp(D - np.max(D))
        if weights.sum() <= 0:
            return 0.0
        p_new = weights / weights.sum()
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    qy = p @ P
    log_ratio = np.zeros_like(P)
    mask = P > 0
    log_qy = np.broadcast_to(np.log(qy + eps), P.shape)
    log_ratio[mask] = np.log(P[mask]) - log_qy[mask]
    I_nats = float(np.sum(p[:, np.newaxis] * P * log_ratio))
    return max(0.0, I_nats / np.log(2.0))


def _greedy_clique_lower_bound(adj_bits: list[int]) -> int:
    """Greedy lower bound for clique size in a graph represented by bitsets."""
    cand = (1 << len(adj_bits)) - 1
    size = 0
    while cand:
        # choose a high-degree candidate inside cand
        verts = [i for i in range(len(adj_bits)) if (cand >> i) & 1]
        v = max(verts, key=lambda i: (adj_bits[i] & cand).bit_count())
        size += 1
        cand &= adj_bits[v]
    return size


def _max_clique_size(adj_bits: list[int], max_exact: int = 80) -> tuple[int, bool]:
    """Maximum clique size.  Exact up to max_exact vertices, greedy otherwise."""
    n = len(adj_bits)
    if n == 0:
        return 0, True
    if n > max_exact:
        return _greedy_clique_lower_bound(adj_bits), False

    best = 0

    def expand(cand: int, size: int) -> None:
        nonlocal best
        if cand == 0:
            if size > best:
                best = size
            return
        if size + cand.bit_count() <= best:
            return
        # branch on vertices, using a simple high-degree ordering heuristic
        while cand:
            if size + cand.bit_count() <= best:
                return
            verts = [i for i in range(n) if (cand >> i) & 1]
            v = max(verts, key=lambda i: (adj_bits[i] & cand).bit_count())
            cand_without_v = cand & ~(1 << v)
            expand(cand_without_v & adj_bits[v], size + 1)
            cand = cand_without_v

    expand((1 << n) - 1, 0)
    return best, True


def zero_error_code_size(
    outputs: dict[int, frozenset[int]],
    max_exact: int = 80,
) -> tuple[int, bool]:
    """One-shot zero-error code size for the set-valued observer channel.

    Two B states are confusable if their possible A-memory sets overlap.  A
    zero-error code is a largest set of pairwise non-confusable B states.
    """
    keys = list(outputs)
    n = len(keys)
    adj_bits = [0 for _ in range(n)]  # non-confusability graph
    sets = [outputs[k] for k in keys]
    for i in range(n):
        si = sets[i]
        for j in range(i + 1, n):
            if si.isdisjoint(sets[j]):
                adj_bits[i] |= 1 << j
                adj_bits[j] |= 1 << i
    return _max_clique_size(adj_bits, max_exact=max_exact)


def finite_observer_measure(
    co: CoupledObserver,
    horizon: int = 2,
    observer_init: InitMode = "zero",
    max_exact_zero_error: int = 80,
    do_shannon: bool = True,
) -> dict:
    """Measure finite-memory resolvability of B by observer A.

    Returns a dictionary suitable for a pandas row.  The physically meaningful
    information columns are:
        zero_error_bits      worst-case / schedule-robust distinguishability
        shannon_capacity_bits stochastic capacity under uniform schedule noise
    The signature columns are diagnostics only; they classify possible-output
    sets externally and should not be treated as observer memory.
    """
    sm = step_maps_coupled(co)
    b_rec = observed_recurrent_states(co)
    log_rec = float(np.log2(len(b_rec))) if len(b_rec) > 0 else float("nan")
    outputs = possible_observer_memories(
        co, horizon=horizon, observer_init=observer_init, step_maps=sm, b_inputs=b_rec
    )
    signatures = {frozenset(v) for v in outputs.values()}
    sig_bits = float(np.log2(len(signatures))) if signatures else 0.0
    ze_size, ze_exact = zero_error_code_size(outputs, max_exact=max_exact_zero_error)
    ze_bits = float(np.log2(max(1, ze_size)))

    cap_bits = float("nan")
    if do_shannon and len(b_rec) > 0:
        _, P = stochastic_observer_channel(
            co, horizon=horizon, observer_init=observer_init, step_maps=sm, b_inputs=b_rec
        )
        cap_bits = blahut_arimoto_capacity(P)

    mem_bits = co.memory_bits
    cut_bits = horizon * co.boundary_width * float(np.log2(co.q))
    physical_bound = min(log_rec, mem_bits, cut_bits) if len(b_rec) else float("nan")
    return dict(
        kB=co.kB,
        kA=co.kA,
        q=co.q,
        w=co.boundary_width,
        horizon=int(horizon),
        observer_init=observer_init,
        n_rec_B=int(len(b_rec)),
        n_A_states=int(co.nA_states),
        n_joint_states=int(co.n_joint_states),
        n_schedules=int(sm.shape[0]),
        logRecB_bits=log_rec,
        memory_bits=float(mem_bits),
        cut_bits=float(cut_bits),
        physical_bound_bits=float(physical_bound),
        signature_classes=int(len(signatures)),
        signature_bits=sig_bits,
        zero_error_code_size=int(ze_size),
        zero_error_exact=bool(ze_exact),
        zero_error_bits=ze_bits,
        shannon_capacity_bits=float(cap_bits),
        zero_error_over_volume=float(ze_bits / log_rec) if log_rec > 0 else float("nan"),
        capacity_over_volume=float(cap_bits / log_rec) if log_rec > 0 else float("nan"),
        capacity_per_boundary_symbol=float(cap_bits / (co.boundary_width * np.log2(co.q)))
        if co.boundary_width > 0 and np.isfinite(cap_bits)
        else float("nan"),
        nontrivial_sccs=int(nontrivial_scc_count(co.joint)),
        interface=str(co.interface),
    )


def run_finite_observer_sweep(
    ks_B: Iterable[int] = (2, 3, 4),
    ks_A: Iterable[int] = (2, 3),
    ws: Iterable[int] = (1,),
    horizons: Iterable[int] = (1, 2, 3),
    q: int = 4,
    n_instances: int = 20,
    observer_init: InitMode = "zero",
    extra_B: int | None = None,
    extra_A: int | None = None,
    base_seed: int = 0,
    max_joint_states: int = 4 ** 7,
    do_shannon: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Grid sweep for finite observer channels.

    The default grid stays small enough for exact enumeration.  Increase
    max_joint_states deliberately; total state count is q**(kB+kA).
    """
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
                    seed = (base_seed * 104729 + kB * 8191 + kA * 1543 + w * 137 + inst) % 2**32
                    rng = np.random.default_rng(seed)
                    co = make_coupled_observer(
                        kB, kA, q, w, rng, extra_B=extra_B, extra_A=extra_A
                    )
                    for horizon in horizons:
                        m = finite_observer_measure(
                            co,
                            horizon=int(horizon),
                            observer_init=observer_init,
                            do_shannon=do_shannon,
                        )
                        m.update(seed=int(seed), extra_B=co.meta["extra_B"], extra_A=co.meta["extra_A"])
                        rows.append(m)
            if verbose:
                print(f"finite-observer kB={kB}, kA={kA} done", flush=True)
    return pd.DataFrame(rows)


def analyze_finite_observer(df: pd.DataFrame) -> dict:
    """Compact verdict for the finite-observer correction."""
    if df.empty:
        return {"verdict": "NO DATA"}
    eps = 1e-8
    mem_viol = int((df["shannon_capacity_bits"] > df["memory_bits"] + eps).sum())
    bound_viol = int((df["shannon_capacity_bits"] > df["physical_bound_bits"] + 1e-6).sum())
    ze_mem_viol = int((df["zero_error_bits"] > df["memory_bits"] + eps).sum())
    group_cols = ["kA", "w", "horizon"]
    scaling = []
    for keys, sub in df.groupby(group_cols):
        by_k = sub.groupby("kB").agg(
            capacity=("shannon_capacity_bits", "mean"),
            zero_error=("zero_error_bits", "mean"),
            volume=("logRecB_bits", "mean"),
            memory=("memory_bits", "mean"),
        )
        if len(by_k) >= 2:
            cap_growth = float(by_k.capacity.iloc[-1] - by_k.capacity.iloc[0])
            vol_growth = float(by_k.volume.iloc[-1] - by_k.volume.iloc[0])
            ratio = cap_growth / vol_growth if abs(vol_growth) > eps else float("nan")
        else:
            ratio = float("nan")
        kA, w, horizon = keys
        scaling.append(
            dict(
                kA=int(kA),
                w=int(w),
                horizon=int(horizon),
                capacity_by_kB={int(k): float(v) for k, v in by_k.capacity.items()},
                zero_error_by_kB={int(k): float(v) for k, v in by_k.zero_error.items()},
                logRec_by_kB={int(k): float(v) for k, v in by_k.volume.items()},
                capacity_growth_over_volume_growth=float(ratio),
            )
        )
    ratios = [s["capacity_growth_over_volume_growth"] for s in scaling if np.isfinite(s["capacity_growth_over_volume_growth"])]
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    if mem_viol or ze_mem_viol or bound_viol:
        verdict = "IMPLEMENTATION WARNING: a capacity bound was violated"
    elif ratios and mean_ratio < 0.35:
        verdict = "FINITE-OBSERVER BOTTLENECK: capacity does not track observed bulk volume"
    elif ratios and mean_ratio > 0.65:
        verdict = "VOLUME-LIKE EVEN FOR FINITE OBSERVER in this grid"
    else:
        verdict = "INCONCLUSIVE / MIXED: increase k-grid, horizons, or instances"
    return dict(
        verdict=verdict,
        memory_bound_violations=mem_viol,
        zero_error_memory_bound_violations=ze_mem_viol,
        physical_bound_violations=bound_viol,
        mean_capacity_growth_over_volume_growth=mean_ratio,
        scaling=scaling,
    )


def plot_finite_observer(df: pd.DataFrame, path: str) -> None:
    """Write a compact diagnostic plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    # One figure, no style choices, no custom colors: compatible with artifact rules.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (kA, w, horizon), sub in df.groupby(["kA", "w", "horizon"]):
        g = sub.groupby("kB").agg(
            capacity=("shannon_capacity_bits", "mean"),
            volume=("logRecB_bits", "mean"),
        )
        label = f"C: kA={kA}, w={w}, T={horizon}"
        ax.plot(g.index, g.capacity, "o-", label=label)
        ax.plot(g.index, g.volume, "s--", label=f"logRec: kA={kA}, T={horizon}")
    ax.set_xlabel("observed bulk size kB")
    ax.set_ylabel("bits")
    ax.set_title("Finite observer capacity vs observed bulk volume")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Finite SCC observer channel test")
    parser.add_argument("q", nargs="?", type=int, default=4)
    parser.add_argument("--ks-B", default="2,3,4", help="comma list, observed SCC sizes")
    parser.add_argument("--ks-A", default="2,3", help="comma list, observer SCC sizes")
    parser.add_argument("--ws", default="1", help="comma list, interface widths")
    parser.add_argument("--horizons", default="1,2,3", help="comma list of tick horizons")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--observer-init", choices=["zero", "all"], default="zero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-joint-states", type=int, default=4 ** 7)
    parser.add_argument("--no-shannon", action="store_true", help="skip Blahut-Arimoto capacity")
    parser.add_argument("--out", default="example_results/finite_observer.csv")
    parser.add_argument("--plot", default="example_results/fig_finite_observer.png")
    args = parser.parse_args(argv)

    df = run_finite_observer_sweep(
        ks_B=_parse_ints(args.ks_B),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        horizons=_parse_ints(args.horizons),
        q=args.q,
        n_instances=args.instances,
        observer_init=args.observer_init,
        base_seed=args.seed,
        max_joint_states=args.max_joint_states,
        do_shannon=not args.no_shannon,
        verbose=True,
    )
    out_path = args.out
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    res = analyze_finite_observer(df)
    summary_path = os.path.join(os.path.dirname(out_path) or ".", "finite_observer_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    try:
        plot_finite_observer(df, args.plot)
    except Exception as exc:  # plotting should not invalidate the numerical run
        print(f"plot skipped: {exc}")
    print(json.dumps(res, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
