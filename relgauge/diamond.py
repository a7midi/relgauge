r"""
diamond.py -- phase-consistency constraints in observer diamonds.

This experiment comes after phasechannel.py.

phasechannel.py asked whether a finite observer A can directly retain another
observer B's schedule phase through a chain B -> A.  In random finite SCCs that
phase was mostly hidden/forgotten.

diamond.py asks a different question: can a downstream consistency condition in
a diamond correlate otherwise independent branch phases?

Diamond structure:

        B
       / \
      C   D
       \ /
        A

B, C, D, and A are distinct SCC observers.  B updates first, C and D are same-
depth branches and may be scheduled in either order, and A updates after both.
Each SCC has arbitrary internal schedule phase.  We impose phase labels on the
branches C and D and measure whether a downstream herald/consistency predicate
selects correlated phase pairs.

Main measured object:

    P(s = 1 | Phi_C, Phi_D)

where s is a finite consistency predicate, usually equality of two paired panels
inside the sink observer A.  With a uniform prior over branch phase labels, we
condition on s=1 and compute

    I(Phi_C ; Phi_D | s=1).

A nonzero value means the diamond consistency predicate induces a relational
constraint between branch phases.  This is deliberately NOT a transmission
experiment.  It is a finite post-selected consistency experiment, so the output
also reports herald efficiency.  A high mutual information at tiny efficiency is
flagged as post-selection-like and should not be overinterpreted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import core as C

PhaseMode = Literal["cut", "erased_value", "cut_value"]
PhaseProtocol = Literal["first", "same"]
InitialMode = Literal["all", "rec"]
ObserverInit = Literal["zero", "all"]
HeraldMode = Literal["sink_equal", "branch_equal"]
BranchOrder = Literal["both", "CD", "DC"]


# --------------------------------------------------------------------------- #
# Diamond construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiamondSystem:
    """Four-SCC feed-forward diamond B -> {C,D} -> A.

    Vertex layout in the joint RelationalSystem:
        B: 0 .. kB-1
        C: offC .. offC+kC-1
        D: offD .. offD+kD-1
        A: offA .. offA+kA-1

    For sink_equal heralds, A has two visible panels of width w:
        A[0:w] receives C-boundary inputs
        A[w:2w] receives D-boundary inputs
    and the herald is A[j] == A[w+j] for all j.
    """

    B: C.RelationalSystem
    Csys: C.RelationalSystem
    Dsys: C.RelationalSystem
    Askel: C.RelationalSystem
    joint: C.RelationalSystem
    kB: int
    kC: int
    kD: int
    kA: int
    q: int
    w: int
    offB: int
    offC: int
    offD: int
    offA: int
    interface_BC: tuple[tuple[int, int], ...]
    interface_BD: tuple[tuple[int, int], ...]
    interface_CA: tuple[tuple[int, int], ...]
    interface_DA: tuple[tuple[int, int], ...]
    meta: dict

    @property
    def k_total(self) -> int:
        return self.kB + self.kC + self.kD + self.kA

    @property
    def n_joint_states(self) -> int:
        return self.q ** self.k_total

    @property
    def sink_memory_bits(self) -> float:
        return self.kA * float(np.log2(self.q))

    @property
    def cut_bits_per_tick(self) -> float:
        # Two branch-to-sink panels of width w are available to the sink.
        return 2.0 * self.w * float(np.log2(self.q))


def _random_rule(q: int, indeg: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, q, size=q ** indeg, dtype=np.int64)


def _edge_adj(sys: C.RelationalSystem) -> dict[int, set[int]]:
    adj = {v: set() for v in range(sys.k)}
    for v in range(sys.k):
        for p in sys.preds[v]:
            adj[p].add(v)
    return adj


def nontrivial_scc_count(sys: C.RelationalSystem) -> int:
    comps = C.tarjan_scc(_edge_adj(sys), range(sys.k))
    return sum(1 for comp in comps if len(comp) >= 2)


def make_diamond_system(
    kB: int,
    kC: int,
    kD: int,
    kA: int,
    q: int,
    w: int,
    rng: np.random.Generator,
    extra_B: int | None = None,
    extra_C: int | None = None,
    extra_D: int | None = None,
    extra_A: int | None = None,
) -> DiamondSystem:
    """Construct B -> {C,D} -> A as four distinct SCC observers.

    The default extra-edge count is one extra edge per vertex in each SCC.  For
    sink_equal, kA must satisfy kA >= 2*w because the sink keeps separate C and
    D panels before the equality herald is evaluated.
    """
    if min(kB, kC, kD, kA) < 1:
        raise ValueError("all component sizes must be positive")
    if q < 2:
        raise ValueError("q must be at least 2")
    w = int(w)
    if w <= 0:
        raise ValueError("w must be positive")
    if w > min(kB, kC, kD):
        raise ValueError("w must be <= min(kB,kC,kD)")
    if kA < 2 * w:
        raise ValueError("sink_equal paneling requires kA >= 2*w")

    if extra_B is None:
        extra_B = kB
    if extra_C is None:
        extra_C = kC
    if extra_D is None:
        extra_D = kD
    if extra_A is None:
        extra_A = kA

    B = C.make_random_scc(kB, q, int(extra_B), rng)
    Cskel = C.make_random_scc(kC, q, int(extra_C), rng)
    Dskel = C.make_random_scc(kD, q, int(extra_D), rng)
    Askel = C.make_random_scc(kA, q, int(extra_A), rng)

    offB = 0
    offC = kB
    offD = kB + kC
    offA = kB + kC + kD
    k_total = kB + kC + kD + kA

    pred_sets: list[set[int]] = [set() for _ in range(k_total)]
    # Internal SCC skeletons.
    for v in range(kB):
        pred_sets[offB + v].update(offB + p for p in B.preds[v])
    for v in range(kC):
        pred_sets[offC + v].update(offC + p for p in Cskel.preds[v])
    for v in range(kD):
        pred_sets[offD + v].update(offD + p for p in Dskel.preds[v])
    for v in range(kA):
        pred_sets[offA + v].update(offA + p for p in Askel.preds[v])

    interface_BC: list[tuple[int, int]] = []
    interface_BD: list[tuple[int, int]] = []
    interface_CA: list[tuple[int, int]] = []
    interface_DA: list[tuple[int, int]] = []

    for j in range(w):
        # B feeds paired boundary vertices on each branch.
        b_src = offB + j
        c_tgt = offC + j
        d_tgt = offD + j
        pred_sets[c_tgt].add(b_src)
        pred_sets[d_tgt].add(b_src)
        interface_BC.append((b_src, c_tgt))
        interface_BD.append((b_src, d_tgt))

        # Branches feed separate panels in A: C -> A[j], D -> A[w+j].
        c_src = offC + j
        d_src = offD + j
        a_c_tgt = offA + j
        a_d_tgt = offA + w + j
        pred_sets[a_c_tgt].add(c_src)
        pred_sets[a_d_tgt].add(d_src)
        interface_CA.append((c_src, a_c_tgt))
        interface_DA.append((d_src, a_d_tgt))

    preds = [tuple(sorted(s)) for s in pred_sets]
    rules: list[np.ndarray] = []
    for v in range(k_total):
        if offB <= v < offC:
            rules.append(B.rules[v - offB].copy())
        else:
            # Branch and sink rules are regenerated after adding external inputs.
            rules.append(_random_rule(q, len(preds[v]), rng))

    joint = C.RelationalSystem(
        k_total,
        q,
        preds,
        rules,
        meta={
            "ensemble": "diamond",
            "kB": kB,
            "kC": kC,
            "kD": kD,
            "kA": kA,
            "w": w,
            "extra_B": int(extra_B),
            "extra_C": int(extra_C),
            "extra_D": int(extra_D),
            "extra_A": int(extra_A),
        },
    )
    return DiamondSystem(
        B=B,
        Csys=Cskel,
        Dsys=Dskel,
        Askel=Askel,
        joint=joint,
        kB=kB,
        kC=kC,
        kD=kD,
        kA=kA,
        q=q,
        w=w,
        offB=offB,
        offC=offC,
        offD=offD,
        offA=offA,
        interface_BC=tuple(interface_BC),
        interface_BD=tuple(interface_BD),
        interface_CA=tuple(interface_CA),
        interface_DA=tuple(interface_DA),
        meta=dict(joint.meta),
    )


# --------------------------------------------------------------------------- #
# Schedules and step maps
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScheduleInfo:
    schedule: tuple[int, ...]
    c_first: int
    d_first: int
    branch_order: str  # "CD" or "DC"


def _first_local(schedule: tuple[int, ...], offset: int, k: int) -> int:
    for v in schedule:
        if offset <= v < offset + k:
            return int(v - offset)
    raise ValueError("component vertices absent from schedule")


def admissible_diamond_schedules(ds: DiamondSystem) -> list[ScheduleInfo]:
    """All block-topological schedules: B, then C/D in either order, then A."""
    Bv = tuple(range(ds.offB, ds.offC))
    Cv = tuple(range(ds.offC, ds.offD))
    Dv = tuple(range(ds.offD, ds.offA))
    Av = tuple(range(ds.offA, ds.offA + ds.kA))
    out: list[ScheduleInfo] = []
    for pB in itertools.permutations(Bv):
        for pC in itertools.permutations(Cv):
            for pD in itertools.permutations(Dv):
                for pA in itertools.permutations(Av):
                    sched_cd = tuple(pB + pC + pD + pA)
                    out.append(
                        ScheduleInfo(
                            schedule=sched_cd,
                            c_first=_first_local(sched_cd, ds.offC, ds.kC),
                            d_first=_first_local(sched_cd, ds.offD, ds.kD),
                            branch_order="CD",
                        )
                    )
                    sched_dc = tuple(pB + pD + pC + pA)
                    out.append(
                        ScheduleInfo(
                            schedule=sched_dc,
                            c_first=_first_local(sched_dc, ds.offC, ds.kC),
                            d_first=_first_local(sched_dc, ds.offD, ds.kD),
                            branch_order="DC",
                        )
                    )
    return out


def step_maps_diamond(ds: DiamondSystem) -> tuple[np.ndarray, list[ScheduleInfo]]:
    infos = admissible_diamond_schedules(ds)
    out = np.empty((len(infos), ds.n_joint_states), dtype=np.int64)
    for i, info in enumerate(infos):
        out[i] = ds.joint.step_map(info.schedule)
    return out, infos


def schedule_indices(
    infos: list[ScheduleInfo],
    c_cut: int | None = None,
    d_cut: int | None = None,
    branch_order: BranchOrder = "both",
) -> np.ndarray:
    idx: list[int] = []
    for i, info in enumerate(infos):
        if c_cut is not None and info.c_first != int(c_cut):
            continue
        if d_cut is not None and info.d_first != int(d_cut):
            continue
        if branch_order != "both" and info.branch_order != branch_order:
            continue
        idx.append(i)
    return np.asarray(idx, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Component-state helpers and phase labels
# --------------------------------------------------------------------------- #
def _component_states(sys: C.RelationalSystem, mode: InitialMode) -> np.ndarray:
    if mode == "all":
        return np.arange(sys.q ** sys.k, dtype=np.int64)
    if mode == "rec":
        sm = sys.all_step_maps()
        adj = C.orbit_adjacency_fast(sm)
        rec = C.recurrent_states(adj)
        if len(rec) == 0:
            return np.arange(sys.q ** sys.k, dtype=np.int64)
        return rec.astype(np.int64)
    raise ValueError("initial mode must be 'all' or 'rec'")


def _digit(indices: np.ndarray, q: int, vertex: int) -> np.ndarray:
    return (np.asarray(indices, dtype=np.int64) // (q ** int(vertex))) % q


def _available_labels(states: np.ndarray, k: int, q: int, mode: PhaseMode, phase_vertex: int) -> tuple:
    if mode == "cut":
        return tuple(range(k))
    if mode == "erased_value":
        vals = sorted(set(_digit(states, q, phase_vertex).astype(int).tolist()))
        return tuple(vals)
    if mode == "cut_value":
        labels = []
        for cut in range(k):
            vals = sorted(set(_digit(states, q, cut).astype(int).tolist()))
            labels.extend((cut, val) for val in vals)
        return tuple(labels)
    raise ValueError(f"unknown phase mode: {mode!r}")


def _states_for_label(
    states: np.ndarray,
    q: int,
    label,
    mode: PhaseMode,
    phase_vertex: int,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64)
    if mode == "cut":
        return states
    if mode == "erased_value":
        return states[_digit(states, q, phase_vertex) == int(label)]
    if mode == "cut_value":
        cut, value = label
        return states[_digit(states, q, int(cut)) == int(value)]
    raise ValueError(f"unknown phase mode: {mode!r}")


def _cut_for_label(label, mode: PhaseMode, phase_vertex: int) -> int:
    if mode == "cut":
        return int(label)
    if mode == "erased_value":
        return int(phase_vertex)
    if mode == "cut_value":
        return int(label[0])
    raise ValueError(f"unknown phase mode: {mode!r}")


def _a_init_states(ds: DiamondSystem, observer_init: ObserverInit) -> np.ndarray:
    if observer_init == "zero":
        return np.array([0], dtype=np.int64)
    if observer_init == "all":
        return np.arange(ds.q ** ds.kA, dtype=np.int64)
    raise ValueError("observer_init must be 'zero' or 'all'")


def _joint_index(ds: DiamondSystem, b: np.ndarray, c: np.ndarray, d: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (
        np.asarray(b, dtype=np.int64)
        + np.asarray(c, dtype=np.int64) * (ds.q ** ds.offC)
        + np.asarray(d, dtype=np.int64) * (ds.q ** ds.offD)
        + np.asarray(a, dtype=np.int64) * (ds.q ** ds.offA)
    )


def _cartesian_joint_indices(
    ds: DiamondSystem,
    b_states: np.ndarray,
    c_states: np.ndarray,
    d_states: np.ndarray,
    a_states: np.ndarray,
) -> np.ndarray:
    # Small exact grids only; use repeat/tile instead of Python loops.
    b_states = np.asarray(b_states, dtype=np.int64)
    c_states = np.asarray(c_states, dtype=np.int64)
    d_states = np.asarray(d_states, dtype=np.int64)
    a_states = np.asarray(a_states, dtype=np.int64)
    nb, nc, nd, na = len(b_states), len(c_states), len(d_states), len(a_states)
    b = np.repeat(b_states, nc * nd * na)
    c = np.tile(np.repeat(c_states, nd * na), nb)
    d = np.tile(np.repeat(d_states, na), nb * nc)
    a = np.tile(a_states, nb * nc * nd)
    return _joint_index(ds, b, c, d, a).astype(np.int64)


# --------------------------------------------------------------------------- #
# Propagation and heralds
# --------------------------------------------------------------------------- #
def _advance_distribution(dist: np.ndarray, step_maps_subset: np.ndarray) -> np.ndarray:
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
    if len(states) == 0:
        return states
    return np.unique(step_maps_subset[:, states].reshape(-1)).astype(np.int64)


def herald_indicator(ds: DiamondSystem, mode: HeraldMode = "sink_equal") -> np.ndarray:
    """Boolean vector over joint states for the declared consistency predicate."""
    states = C.all_states(ds.k_total, ds.q)
    if mode == "sink_equal":
        left = states[:, ds.offA : ds.offA + ds.w]
        right = states[:, ds.offA + ds.w : ds.offA + 2 * ds.w]
        return np.all(left == right, axis=1)
    if mode == "branch_equal":
        left = states[:, ds.offC : ds.offC + ds.w]
        right = states[:, ds.offD : ds.offD + ds.w]
        return np.all(left == right, axis=1)
    raise ValueError("herald mode must be 'sink_equal' or 'branch_equal'")


@dataclass(frozen=True)
class DiamondChannel:
    labels_C: tuple
    labels_D: tuple
    P_herald: np.ndarray             # P(s=1 | phi_C, phi_D)
    possible_herald: np.ndarray      # True if s=1 is possible for pair
    certain_herald: np.ndarray       # True if s=1 for all possible outcomes
    mode: str
    protocol: str
    herald_mode: str
    phase_vertex_C: int
    phase_vertex_D: int


def diamond_herald_matrix(
    ds: DiamondSystem,
    horizon: int,
    phase_mode: PhaseMode = "erased_value",
    protocol: PhaseProtocol = "first",
    herald_mode: HeraldMode = "sink_equal",
    branch_order: BranchOrder = "both",
    initial_mode: InitialMode = "all",
    observer_init: ObserverInit = "zero",
    phase_vertex_C: int = 0,
    phase_vertex_D: int = 0,
    step_maps: np.ndarray | None = None,
    infos: list[ScheduleInfo] | None = None,
    do_robust: bool = True,
) -> DiamondChannel:
    """Exact matrix P(s=1 | Phi_C, Phi_D)."""
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    if protocol not in ("first", "same"):
        raise ValueError("protocol must be 'first' or 'same'")
    if branch_order not in ("both", "CD", "DC"):
        raise ValueError("branch_order must be 'both', 'CD', or 'DC'")
    if step_maps is None or infos is None:
        step_maps, infos = step_maps_diamond(ds)

    all_idx = schedule_indices(infos, branch_order=branch_order)
    if len(all_idx) == 0:
        raise ValueError("no admissible schedules after branch_order restriction")

    B_states = _component_states(ds.B, initial_mode)
    C_states_all = _component_states(ds.Csys, initial_mode)
    D_states_all = _component_states(ds.Dsys, initial_mode)
    A_states = _a_init_states(ds, observer_init)
    labels_C = _available_labels(C_states_all, ds.kC, ds.q, phase_mode, phase_vertex_C)
    labels_D = _available_labels(D_states_all, ds.kD, ds.q, phase_mode, phase_vertex_D)
    H = herald_indicator(ds, herald_mode)

    P = np.zeros((len(labels_C), len(labels_D)), dtype=float)
    possible = np.zeros_like(P, dtype=bool)
    certain = np.zeros_like(P, dtype=bool)

    for i, labC in enumerate(labels_C):
        c_cut = _cut_for_label(labC, phase_mode, phase_vertex_C)
        C_states = _states_for_label(C_states_all, ds.q, labC, phase_mode, phase_vertex_C)
        for j, labD in enumerate(labels_D):
            d_cut = _cut_for_label(labD, phase_mode, phase_vertex_D)
            D_states = _states_for_label(D_states_all, ds.q, labD, phase_mode, phase_vertex_D)
            first_idx = schedule_indices(infos, c_cut=c_cut, d_cut=d_cut, branch_order=branch_order)
            if len(first_idx) == 0 or len(C_states) == 0 or len(D_states) == 0:
                P[i, j] = np.nan
                continue

            init = _cartesian_joint_indices(ds, B_states, C_states, D_states, A_states)
            dist = np.zeros(ds.n_joint_states, dtype=float)
            if len(init):
                vals, counts = np.unique(init, return_counts=True)
                dist[vals] = counts.astype(float) / float(counts.sum())
            for t in range(horizon):
                idx = first_idx if (protocol == "same" or t == 0) else all_idx
                dist = _advance_distribution(dist, step_maps[idx])
            P[i, j] = float(dist[H].sum())

            if do_robust:
                states = init
                for t in range(horizon):
                    idx = first_idx if (protocol == "same" or t == 0) else all_idx
                    states = _advance_set(states, step_maps[idx])
                if len(states):
                    vals = H[states]
                    possible[i, j] = bool(np.any(vals))
                    certain[i, j] = bool(np.all(vals))

    return DiamondChannel(
        labels_C=tuple(labels_C),
        labels_D=tuple(labels_D),
        P_herald=P,
        possible_herald=possible,
        certain_herald=certain,
        mode=phase_mode,
        protocol=protocol,
        herald_mode=herald_mode,
        phase_vertex_C=int(phase_vertex_C),
        phase_vertex_D=int(phase_vertex_D),
    )


# --------------------------------------------------------------------------- #
# Information analysis
# --------------------------------------------------------------------------- #
def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _kl_to_uniform(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    n = len(p)
    if n == 0:
        return 0.0
    return float(np.sum(p * (np.log2(p) + np.log2(n))))


def _posterior_metrics(P_herald: np.ndarray) -> dict:
    H = np.asarray(P_herald, dtype=float)
    mask = np.isfinite(H)
    if not np.any(mask):
        return dict(
            herald_efficiency=float("nan"),
            posterior_entropy_bits=float("nan"),
            selection_bits=float("nan"),
            marginal_selection_C_bits=float("nan"),
            marginal_selection_D_bits=float("nan"),
            relation_mi_bits=float("nan"),
            relation_mi_norm=float("nan"),
            herald_spread=float("nan"),
            herald_cv=float("nan"),
        )
    H2 = np.where(mask, np.clip(H, 0.0, 1.0), 0.0)
    nC, nD = H2.shape
    n_pairs = max(1, nC * nD)
    eff = float(H2.sum() / n_pairs)
    finite_vals = H[mask]
    spread = float(np.nanmax(finite_vals) - np.nanmin(finite_vals)) if len(finite_vals) else float("nan")
    mean = float(np.nanmean(finite_vals)) if len(finite_vals) else float("nan")
    std = float(np.nanstd(finite_vals)) if len(finite_vals) else float("nan")
    cv = float(std / mean) if mean > 1e-12 else float("nan")
    if eff <= 0:
        return dict(
            herald_efficiency=0.0,
            posterior_entropy_bits=0.0,
            selection_bits=float(np.log2(n_pairs)),
            marginal_selection_C_bits=float("nan"),
            marginal_selection_D_bits=float("nan"),
            relation_mi_bits=float("nan"),
            relation_mi_norm=float("nan"),
            herald_spread=spread,
            herald_cv=cv,
        )
    post = H2 / H2.sum()
    pC = post.sum(axis=1)
    pD = post.sum(axis=0)
    post_entropy = _entropy(post.reshape(-1))
    selection = float(np.log2(n_pairs) - post_entropy)
    selC = _kl_to_uniform(pC)
    selD = _kl_to_uniform(pD)
    mi = 0.0
    for i in range(nC):
        for j in range(nD):
            p = post[i, j]
            if p > 0 and pC[i] > 0 and pD[j] > 0:
                mi += float(p * np.log2(p / (pC[i] * pD[j])))
    denom = min(np.log2(max(1, nC)), np.log2(max(1, nD)))
    norm = float(mi / denom) if denom > 0 else 0.0
    return dict(
        herald_efficiency=eff,
        posterior_entropy_bits=float(post_entropy),
        selection_bits=selection,
        marginal_selection_C_bits=selC,
        marginal_selection_D_bits=selD,
        relation_mi_bits=float(mi),
        relation_mi_norm=norm,
        herald_spread=spread,
        herald_cv=cv,
    )


def diamond_measure(
    ds: DiamondSystem,
    horizon: int,
    phase_mode: PhaseMode = "erased_value",
    protocol: PhaseProtocol = "first",
    herald_mode: HeraldMode = "sink_equal",
    branch_order: BranchOrder = "both",
    initial_mode: InitialMode = "all",
    observer_init: ObserverInit = "zero",
    phase_vertex_C: int = 0,
    phase_vertex_D: int = 0,
    do_robust: bool = True,
    step_maps: np.ndarray | None = None,
    infos: list[ScheduleInfo] | None = None,
) -> dict:
    if step_maps is None or infos is None:
        sm, infos = step_maps_diamond(ds)
    else:
        sm = step_maps
    ch = diamond_herald_matrix(
        ds,
        horizon=horizon,
        phase_mode=phase_mode,
        protocol=protocol,
        herald_mode=herald_mode,
        branch_order=branch_order,
        initial_mode=initial_mode,
        observer_init=observer_init,
        phase_vertex_C=phase_vertex_C,
        phase_vertex_D=phase_vertex_D,
        step_maps=sm,
        infos=infos,
        do_robust=do_robust,
    )
    metrics = _posterior_metrics(ch.P_herald)
    nC = len(ch.labels_C)
    nD = len(ch.labels_D)
    phase_entropy_C = float(np.log2(nC)) if nC > 0 else float("nan")
    phase_entropy_D = float(np.log2(nD)) if nD > 0 else float("nan")
    pair_entropy = float(np.log2(max(1, nC * nD)))
    robust_possible_frac = float(np.mean(ch.possible_herald)) if ch.possible_herald.size else float("nan")
    robust_certain_frac = float(np.mean(ch.certain_herald)) if ch.certain_herald.size else float("nan")

    return dict(
        kB=ds.kB,
        kC=ds.kC,
        kD=ds.kD,
        kA=ds.kA,
        q=ds.q,
        w=ds.w,
        horizon=int(horizon),
        phase_mode=phase_mode,
        phase_protocol=protocol,
        herald_mode=herald_mode,
        branch_order=branch_order,
        initial_mode=initial_mode,
        observer_init=observer_init,
        phase_vertex_C=int(phase_vertex_C),
        phase_vertex_D=int(phase_vertex_D),
        n_phase_C=int(nC),
        n_phase_D=int(nD),
        phase_entropy_C_bits=phase_entropy_C,
        phase_entropy_D_bits=phase_entropy_D,
        phase_pair_entropy_bits=pair_entropy,
        relation_mi_bits=float(metrics["relation_mi_bits"]),
        relation_mi_norm=float(metrics["relation_mi_norm"]),
        selection_bits=float(metrics["selection_bits"]),
        posterior_entropy_bits=float(metrics["posterior_entropy_bits"]),
        marginal_selection_C_bits=float(metrics["marginal_selection_C_bits"]),
        marginal_selection_D_bits=float(metrics["marginal_selection_D_bits"]),
        herald_efficiency=float(metrics["herald_efficiency"]),
        herald_spread=float(metrics["herald_spread"]),
        herald_cv=float(metrics["herald_cv"]),
        robust_possible_fraction=robust_possible_frac,
        robust_certain_fraction=robust_certain_frac,
        sink_memory_bits=float(ds.sink_memory_bits),
        cut_bits_per_tick=float(ds.cut_bits_per_tick),
        n_joint_states=int(ds.n_joint_states),
        n_schedules=int(sm.shape[0]),
        nontrivial_sccs=int(nontrivial_scc_count(ds.joint)),
        interface_BC=str(ds.interface_BC),
        interface_BD=str(ds.interface_BD),
        interface_CA=str(ds.interface_CA),
        interface_DA=str(ds.interface_DA),
        labels_C=str(ch.labels_C),
        labels_D=str(ch.labels_D),
        herald_matrix=json.dumps(ch.P_herald.tolist()),
    )


# --------------------------------------------------------------------------- #
# Sweeps / analysis / plotting
# --------------------------------------------------------------------------- #
def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(text).split(",") if x.strip())


def _parse_modes(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def run_diamond_sweep(
    ks_B: Iterable[int] = (2,),
    ks_C: Iterable[int] = (2,),
    ks_D: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2,),
    ws: Iterable[int] = (1,),
    horizons: Iterable[int] = (1, 2, 3),
    q: int = 4,
    n_instances: int = 20,
    phase_modes: Iterable[PhaseMode] = ("cut", "erased_value"),
    protocols: Iterable[PhaseProtocol] = ("first",),
    herald_modes: Iterable[HeraldMode] = ("sink_equal",),
    branch_orders: Iterable[BranchOrder] = ("both",),
    initial_mode: InitialMode = "all",
    observer_init: ObserverInit = "zero",
    phase_vertex_C: int = 0,
    phase_vertex_D: int = 0,
    extra_B: int | None = None,
    extra_C: int | None = None,
    extra_D: int | None = None,
    extra_A: int | None = None,
    base_seed: int = 0,
    max_joint_states: int = 4 ** 8,
    do_robust: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for kB in ks_B:
        for kC in ks_C:
            for kD in ks_D:
                for kA in ks_A:
                    if q ** (kB + kC + kD + kA) > max_joint_states:
                        if verbose:
                            print(f"skip kB={kB},kC={kC},kD={kD},kA={kA}: q^K too large", flush=True)
                        continue
                    for w in ws:
                        if w > min(kB, kC, kD) or kA < 2 * w:
                            if verbose:
                                print(f"skip w={w} for kA={kA}: needs w<=min branches and kA>=2w", flush=True)
                            continue
                        for inst in range(n_instances):
                            seed = (
                                base_seed * 1000003
                                + kB * 92821
                                + kC * 68917
                                + kD * 51721
                                + kA * 3571
                                + w * 809
                                + inst
                            ) % 2**32
                            rng = np.random.default_rng(seed)
                            ds = make_diamond_system(
                                kB,
                                kC,
                                kD,
                                kA,
                                q,
                                w,
                                rng,
                                extra_B=extra_B,
                                extra_C=extra_C,
                                extra_D=extra_D,
                                extra_A=extra_A,
                            )
                            sm, infos = step_maps_diamond(ds)
                            for horizon in horizons:
                                for phase_mode in phase_modes:
                                    for protocol in protocols:
                                        for herald_mode in herald_modes:
                                            for branch_order in branch_orders:
                                                m = diamond_measure(
                                                    ds,
                                                    horizon=int(horizon),
                                                    phase_mode=phase_mode,  # type: ignore[arg-type]
                                                    protocol=protocol,  # type: ignore[arg-type]
                                                    herald_mode=herald_mode,  # type: ignore[arg-type]
                                                    branch_order=branch_order,  # type: ignore[arg-type]
                                                    initial_mode=initial_mode,
                                                    observer_init=observer_init,
                                                    phase_vertex_C=phase_vertex_C,
                                                    phase_vertex_D=phase_vertex_D,
                                                    do_robust=do_robust,
                                                    step_maps=sm,
                                                    infos=infos,
                                                )
                                                m.update(
                                                    seed=int(seed),
                                                    instance=int(inst),
                                                    extra_B=ds.meta["extra_B"],
                                                    extra_C=ds.meta["extra_C"],
                                                    extra_D=ds.meta["extra_D"],
                                                    extra_A=ds.meta["extra_A"],
                                                )
                                                rows.append(m)
                        if verbose:
                            print(
                                f"diamond kB={kB}, kC={kC}, kD={kD}, kA={kA}, w={w} done",
                                flush=True,
                            )
    return pd.DataFrame(rows)


def analyze_diamond(df: pd.DataFrame, min_efficiency: float = 0.02) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    clean = df.replace([np.inf, -np.inf], np.nan)
    mean_eff = float(clean["herald_efficiency"].mean())
    mean_mi = float(clean["relation_mi_bits"].mean())
    mean_norm = float(clean["relation_mi_norm"].mean())
    mean_sel = float(clean["selection_bits"].mean())
    mean_spread = float(clean["herald_spread"].mean())
    mean_certain = float(clean["robust_certain_fraction"].mean()) if "robust_certain_fraction" in clean else float("nan")

    group_cols = ["phase_mode", "phase_protocol", "herald_mode", "branch_order", "w", "horizon"]
    scaling = []
    for keys, sub in clean.groupby(group_cols):
        phase_mode, protocol, herald_mode, branch_order, w, horizon = keys
        scaling.append(
            dict(
                phase_mode=str(phase_mode),
                phase_protocol=str(protocol),
                herald_mode=str(herald_mode),
                branch_order=str(branch_order),
                w=int(w),
                horizon=int(horizon),
                herald_efficiency=float(sub["herald_efficiency"].mean()),
                relation_mi_bits=float(sub["relation_mi_bits"].mean()),
                relation_mi_norm=float(sub["relation_mi_norm"].mean()),
                selection_bits=float(sub["selection_bits"].mean()),
                herald_spread=float(sub["herald_spread"].mean()),
                robust_possible_fraction=float(sub["robust_possible_fraction"].mean()),
                robust_certain_fraction=float(sub["robust_certain_fraction"].mean()),
            )
        )

    low_eff_frac = float((clean["herald_efficiency"] < min_efficiency).mean())
    high_relation = clean[(clean["herald_efficiency"] >= min_efficiency) & (clean["relation_mi_norm"] > 0.10)]
    high_marginal = clean[(clean["herald_efficiency"] >= min_efficiency) & (clean["selection_bits"] > 0.20)]

    if len(high_relation) / max(1, len(clean)) > 0.25:
        verdict = "DIAMOND PHASE-CONSISTENCY: downstream herald correlates branch phases"
    elif mean_eff < min_efficiency and mean_norm > 0.10:
        verdict = "LOW-EFFICIENCY / POST-SELECTION-LIKE phase relation; inspect efficiency"
    elif len(high_marginal) / max(1, len(clean)) > 0.25 and mean_norm <= 0.10:
        verdict = "MARGINAL PHASE FILTERING: consistency selects phases but does not relate branches"
    elif mean_norm <= 0.03 and mean_sel <= 0.10:
        verdict = "NO GENERIC DIAMOND PHASE CONSTRAINT in this regime"
    else:
        verdict = "MIXED / WEAK DIAMOND CONSTRAINT: inspect modes, horizons, and herald matrices"

    return dict(
        verdict=verdict,
        min_efficiency_threshold=float(min_efficiency),
        mean_herald_efficiency=mean_eff,
        low_efficiency_fraction=low_eff_frac,
        mean_relation_mi_bits=mean_mi,
        mean_relation_mi_norm=mean_norm,
        mean_selection_bits=mean_sel,
        mean_herald_spread=mean_spread,
        mean_robust_certain_fraction=mean_certain,
        high_relation_rows=int(len(high_relation)),
        high_marginal_selection_rows=int(len(high_marginal)),
        n_rows=int(len(clean)),
        scaling=scaling,
    )


def plot_diamond(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, protocol, w), sub in df.groupby(["phase_mode", "phase_protocol", "w"]):
        g = sub.groupby("horizon")["relation_mi_norm"].mean().dropna()
        ax.plot(g.index, g.values, "o-", label=f"{mode}, {protocol}, w={w}")
    ax.set_xlabel("horizon T")
    ax.set_ylabel(r"normalized $I(\Phi_C;\Phi_D\mid s=1)$")
    ax.set_title("Diamond phase-consistency relation strength")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diamond phase-consistency experiment")
    parser.add_argument("q", nargs="?", type=int, default=4)
    parser.add_argument("--ks-B", default="2")
    parser.add_argument("--ks-C", default="2")
    parser.add_argument("--ks-D", default="2")
    parser.add_argument("--ks-A", default="2")
    parser.add_argument("--ws", default="1")
    parser.add_argument("--horizons", default="1,2,3")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--phase-modes", default="cut,erased_value", help="cut,erased_value,cut_value")
    parser.add_argument("--protocols", default="first", help="first,same")
    parser.add_argument("--herald-modes", default="sink_equal", help="sink_equal,branch_equal")
    parser.add_argument("--branch-orders", default="both", help="both,CD,DC")
    parser.add_argument("--initial-mode", choices=["all", "rec"], default="all")
    parser.add_argument("--observer-init", choices=["zero", "all"], default="zero")
    parser.add_argument("--phase-vertex-C", type=int, default=0)
    parser.add_argument("--phase-vertex-D", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-joint-states", type=int, default=4 ** 8)
    parser.add_argument("--min-efficiency", type=float, default=0.02)
    parser.add_argument("--no-robust", action="store_true", help="skip possible/certain herald set propagation")
    parser.add_argument("--out", default="example_results/diamond.csv")
    parser.add_argument("--plot", default="example_results/fig_diamond.png")
    args = parser.parse_args(argv)

    allowed_phase = {"cut", "erased_value", "cut_value"}
    allowed_protocol = {"first", "same"}
    allowed_herald = {"sink_equal", "branch_equal"}
    allowed_order = {"both", "CD", "DC"}
    phase_modes = _parse_modes(args.phase_modes)
    protocols = _parse_modes(args.protocols)
    herald_modes = _parse_modes(args.herald_modes)
    branch_orders = _parse_modes(args.branch_orders)
    bad = set(phase_modes) - allowed_phase
    if bad:
        raise SystemExit(f"unknown phase mode(s): {sorted(bad)}")
    bad = set(protocols) - allowed_protocol
    if bad:
        raise SystemExit(f"unknown protocol(s): {sorted(bad)}")
    bad = set(herald_modes) - allowed_herald
    if bad:
        raise SystemExit(f"unknown herald mode(s): {sorted(bad)}")
    bad = set(branch_orders) - allowed_order
    if bad:
        raise SystemExit(f"unknown branch order(s): {sorted(bad)}")

    df = run_diamond_sweep(
        ks_B=_parse_ints(args.ks_B),
        ks_C=_parse_ints(args.ks_C),
        ks_D=_parse_ints(args.ks_D),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        horizons=_parse_ints(args.horizons),
        q=args.q,
        n_instances=args.instances,
        phase_modes=phase_modes,  # type: ignore[arg-type]
        protocols=protocols,  # type: ignore[arg-type]
        herald_modes=herald_modes,  # type: ignore[arg-type]
        branch_orders=branch_orders,  # type: ignore[arg-type]
        initial_mode=args.initial_mode,
        observer_init=args.observer_init,
        phase_vertex_C=args.phase_vertex_C,
        phase_vertex_D=args.phase_vertex_D,
        base_seed=args.seed,
        max_joint_states=args.max_joint_states,
        do_robust=not args.no_robust,
        verbose=True,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    res = analyze_diamond(df, min_efficiency=args.min_efficiency)
    summary_path = os.path.join(os.path.dirname(args.out) or ".", "diamond_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    try:
        plot_diamond(df, args.plot)
    except Exception as exc:
        print(f"plot skipped: {exc}")
    print(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
