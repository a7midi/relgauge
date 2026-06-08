"""
approxequalizer.py -- approximate shared-interface equalizers under noisy rules.

Why this exists
---------------
``interfaceequalizer.py`` computes an exact shared interface alphabet S by taking
connected components of the *support* of the left/right sink-panel compatibility
graph.  That is the correct zero-error object, but it is brittle: a single weak
cross-talk edge connects two components and collapses the exact |S| to 1.

The rule-selection mutation sweep showed precisely this pattern: exact |S|
falls quickly under mutation while interface-rule mutual information remains
well above the random null.  This module therefore computes an approximate
shared alphabet S_eps by discarding weak compatibility edges, recovering strong
components, and measuring how much latent common information survives.

The central continuous diagnostic is

    approx_info_qcoords = I(Z_L ; Z_R) / log2(q),

where Z_L and Z_R are the approximate component labels inferred from the left
and right panels.  This is an approximate-equalizer analogue of log_q |S|.

Interpretation
--------------
* exact_residual_qcoords > 0:
    strict/zero-error interface law.
* approx_residual_qcoords > 0 but exact_residual_qcoords = 0:
    noisy but recoverable latent interface law.
* approx_info_qcoords near random:
    no meaningful shared interface label.

The experiment compares random rules with structured latent-label preserving
rules (copy, permutive, block, canalizing) and mutation rates.  A positive result
means the consistency principle selects not only exact conserved labels, but
also approximate/error-tolerant latent labels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import interfaceequalizer as IE
from . import ruleselection as R
from . import scdcdiamond as S

RuleEnsemble = R.RuleEnsemble
EqualizerMode = IE.EqualizerMode
InitialPool = S.InitialPool
ObserverInit = S.ObserverInit
ThresholdMode = Literal["conditional", "relative", "mass"]


# --------------------------------------------------------------------------- #
# Information helpers
# --------------------------------------------------------------------------- #
def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _mutual_info_from_joint(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=float)
    total = float(joint.sum())
    if total <= 0:
        return 0.0
    p = joint / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    denom = px @ py
    return float((p[nz] * np.log2(p[nz] / denom[nz])).sum())


# --------------------------------------------------------------------------- #
# Compatibility matrix and thresholded equalizer
# --------------------------------------------------------------------------- #
def compatibility_matrix(left_words: np.ndarray, right_words: np.ndarray, q: int, width: int) -> np.ndarray:
    """Return the left/right compatibility count matrix.

    Rows are left-panel words, columns are right-panel words.  The side alphabet
    has size q**width.
    """
    left_words = np.asarray(left_words, dtype=np.int64)
    right_words = np.asarray(right_words, dtype=np.int64)
    if left_words.shape != right_words.shape:
        raise ValueError("left_words and right_words must have the same shape")
    n_side = int(q ** width)
    M = np.zeros((n_side, n_side), dtype=np.int64)
    if len(left_words) == 0:
        return M
    if np.any(left_words < 0) or np.any(left_words >= n_side) or np.any(right_words < 0) or np.any(right_words >= n_side):
        raise ValueError("word outside side alphabet")
    np.add.at(M, (left_words, right_words), 1)
    return M


class _UF:
    def __init__(self, n: int):
        self.p = list(range(int(n)))

    def find(self, x: int) -> int:
        r = int(x)
        while self.p[r] != r:
            r = self.p[r]
        x = int(x)
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def _threshold_edges(M: np.ndarray, eps: float, mode: ThresholdMode = "conditional") -> np.ndarray:
    """Return a boolean matrix of retained strong compatibility edges.

    ``conditional`` keeps an observed edge if it is at least eps-likely from the
    row side and from the column side:

        P(R=r | L=l) >= eps  and  P(L=l | R=r) >= eps.

    For q=3 or 4 and w=1, random complete mixing has conditional probabilities
    near 1/q, so eps=0.15 or 0.2 keeps the random graph connected.  Mutated
    copy/block rules typically have strong diagonal/block edges plus weak
    cross-talk; the weak cross-talk is discarded.

    ``relative`` keeps edges whose count is at least eps times both their row
    maximum and column maximum.  ``mass`` keeps edges whose joint probability is
    at least eps times the largest joint edge probability.
    """
    M = np.asarray(M, dtype=float)
    observed = M > 0
    if M.sum() <= 0:
        return np.zeros_like(M, dtype=bool)
    eps = float(eps)
    if eps <= 0:
        return observed.copy()
    if mode == "conditional":
        row_sums = M.sum(axis=1, keepdims=True)
        col_sums = M.sum(axis=0, keepdims=True)
        row_p = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
        col_p = np.divide(M, col_sums, out=np.zeros_like(M), where=col_sums > 0)
        return observed & (row_p >= eps) & (col_p >= eps)
    if mode == "relative":
        row_max = M.max(axis=1, keepdims=True)
        col_max = M.max(axis=0, keepdims=True)
        return observed & (M >= eps * row_max) & (M >= eps * col_max)
    if mode == "mass":
        mx = float(M.max())
        return observed & (M >= eps * mx)
    raise ValueError(f"unknown threshold mode: {mode!r}")


@dataclass(frozen=True)
class ApproxEqualizerSummary:
    q: int
    width: int
    threshold: float
    threshold_mode: str
    alphabet_size_per_side: int
    n_observed_pairs: int
    exact_shared_classes: int
    exact_residual_qcoords: float
    exact_cost_qcoords: float
    exact_edge_density: float
    approx_classes: int
    approx_residual_qcoords: float
    approx_cost_qcoords: float
    approx_kept_edge_fraction: float
    approx_kept_mass: float
    approx_same_component_mass: float
    approx_cross_component_mass: float
    approx_unassigned_mass: float
    approx_component_mi_bits: float
    approx_info_qcoords: float
    approx_info_over_exact_capacity: float
    approx_component_entropy_left_bits: float
    approx_component_entropy_right_bits: float
    approx_min_component_entropy_bits: float
    approx_component_balance: float
    approx_class_sizes_left: tuple[int, ...]
    approx_class_sizes_right: tuple[int, ...]

    def as_dict(self, prefix: str = "") -> dict:
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}threshold": float(self.threshold),
            f"{p}threshold_mode": str(self.threshold_mode),
            f"{p}alphabet_size_per_side": int(self.alphabet_size_per_side),
            f"{p}n_observed_pairs": int(self.n_observed_pairs),
            f"{p}exact_shared_classes": int(self.exact_shared_classes),
            f"{p}exact_residual_qcoords": float(self.exact_residual_qcoords),
            f"{p}exact_cost_qcoords": float(self.exact_cost_qcoords),
            f"{p}exact_edge_density": float(self.exact_edge_density),
            f"{p}approx_classes": int(self.approx_classes),
            f"{p}approx_residual_qcoords": float(self.approx_residual_qcoords),
            f"{p}approx_cost_qcoords": float(self.approx_cost_qcoords),
            f"{p}approx_kept_edge_fraction": float(self.approx_kept_edge_fraction),
            f"{p}approx_kept_mass": float(self.approx_kept_mass),
            f"{p}approx_same_component_mass": float(self.approx_same_component_mass),
            f"{p}approx_cross_component_mass": float(self.approx_cross_component_mass),
            f"{p}approx_unassigned_mass": float(self.approx_unassigned_mass),
            f"{p}approx_component_mi_bits": float(self.approx_component_mi_bits),
            f"{p}approx_info_qcoords": float(self.approx_info_qcoords),
            f"{p}approx_info_over_exact_capacity": float(self.approx_info_over_exact_capacity),
            f"{p}approx_component_entropy_left_bits": float(self.approx_component_entropy_left_bits),
            f"{p}approx_component_entropy_right_bits": float(self.approx_component_entropy_right_bits),
            f"{p}approx_min_component_entropy_bits": float(self.approx_min_component_entropy_bits),
            f"{p}approx_component_balance": float(self.approx_component_balance),
            f"{p}approx_class_sizes_left": json.dumps([int(x) for x in self.approx_class_sizes_left]),
            f"{p}approx_class_sizes_right": json.dumps([int(x) for x in self.approx_class_sizes_right]),
        }


def approximate_shared_equalizer_from_matrix(
    M: np.ndarray,
    q: int,
    width: int,
    threshold: float = 0.2,
    threshold_mode: ThresholdMode = "conditional",
) -> ApproxEqualizerSummary:
    """Compute exact and thresholded/approximate shared interface alphabets.

    The exact part is the old support-connected-component equalizer.  The
    approximate part thresholds weak edges before forming components, then
    evaluates the resulting component labels against the full, unthresholded
    compatibility distribution.
    """
    M = np.asarray(M, dtype=np.int64)
    n_side = int(q ** width)
    if M.shape != (n_side, n_side):
        raise ValueError(f"M must have shape {(n_side, n_side)}, got {M.shape}")
    total = int(M.sum())
    if total <= 0:
        return ApproxEqualizerSummary(
            q=q,
            width=width,
            threshold=float(threshold),
            threshold_mode=str(threshold_mode),
            alphabet_size_per_side=n_side,
            n_observed_pairs=0,
            exact_shared_classes=0,
            exact_residual_qcoords=math.nan,
            exact_cost_qcoords=math.nan,
            exact_edge_density=0.0,
            approx_classes=0,
            approx_residual_qcoords=math.nan,
            approx_cost_qcoords=math.nan,
            approx_kept_edge_fraction=0.0,
            approx_kept_mass=0.0,
            approx_same_component_mass=0.0,
            approx_cross_component_mass=0.0,
            approx_unassigned_mass=1.0,
            approx_component_mi_bits=0.0,
            approx_info_qcoords=0.0,
            approx_info_over_exact_capacity=0.0,
            approx_component_entropy_left_bits=0.0,
            approx_component_entropy_right_bits=0.0,
            approx_min_component_entropy_bits=0.0,
            approx_component_balance=0.0,
            approx_class_sizes_left=(),
            approx_class_sizes_right=(),
        )

    # Exact equalizer via the existing support graph kernel.
    left, right = np.nonzero(M)
    weights = M[left, right]
    left_rep = np.repeat(left, weights)
    right_rep = np.repeat(right, weights)
    exact = IE.maximal_shared_equalizer(left_rep, right_rep, q, width)

    # Approximate equalizer via thresholded support.
    keep = _threshold_edges(M, float(threshold), threshold_mode)
    kept_pairs = np.argwhere(keep)
    approx_n_edges = int(keep.sum())
    observed_edges = int((M > 0).sum())
    kept_mass = float(M[keep].sum() / total) if total else 0.0

    if approx_n_edges == 0:
        approx_classes = 0
        label_left = np.full(n_side, -1, dtype=np.int64)
        label_right = np.full(n_side, -1, dtype=np.int64)
        class_sizes_left: tuple[int, ...] = ()
        class_sizes_right: tuple[int, ...] = ()
    else:
        uf = _UF(2 * n_side)
        touched_left: set[int] = set()
        touched_right: set[int] = set()
        for l, r in kept_pairs:
            l_i, r_i = int(l), int(r)
            touched_left.add(l_i)
            touched_right.add(r_i)
            uf.union(l_i, n_side + r_i)
        roots = sorted({uf.find(x) for x in touched_left} | {uf.find(n_side + x) for x in touched_right})
        root_to_label = {root: i for i, root in enumerate(roots)}
        approx_classes = len(roots)
        label_left = np.full(n_side, -1, dtype=np.int64)
        label_right = np.full(n_side, -1, dtype=np.int64)
        for x in touched_left:
            label_left[x] = root_to_label[uf.find(x)]
        for x in touched_right:
            label_right[x] = root_to_label[uf.find(n_side + x)]
        class_sizes_left = tuple(int(np.sum(label_left == i)) for i in range(approx_classes))
        class_sizes_right = tuple(int(np.sum(label_right == i)) for i in range(approx_classes))

    residual_q = math.log(approx_classes, q) if approx_classes > 0 else 0.0
    cost_q = 2 * width - residual_q

    # Evaluate component labels on the full distribution, not just retained edges.
    same_mass = 0.0
    cross_mass = 0.0
    unassigned_mass = 0.0
    if approx_classes > 0:
        joint_comp = np.zeros((approx_classes, approx_classes), dtype=float)
        for l in range(n_side):
            for r in range(n_side):
                c = int(M[l, r])
                if c == 0:
                    continue
                zl = int(label_left[l])
                zr = int(label_right[r])
                mass = c / total
                if zl < 0 or zr < 0:
                    unassigned_mass += mass
                else:
                    joint_comp[zl, zr] += c
                    if zl == zr:
                        same_mass += mass
                    else:
                        cross_mass += mass
        mi_bits = _mutual_info_from_joint(joint_comp)
        h_left = _entropy_from_counts(joint_comp.sum(axis=1))
        h_right = _entropy_from_counts(joint_comp.sum(axis=0))
        min_h = min(h_left, h_right)
        # Balance = effective entropy relative to log2(number of classes).
        max_h = math.log2(approx_classes) if approx_classes > 1 else 0.0
        balance = float(min_h / max_h) if max_h > 0 else 0.0
    else:
        mi_bits = 0.0
        h_left = 0.0
        h_right = 0.0
        min_h = 0.0
        balance = 0.0
        unassigned_mass = 1.0

    info_q = mi_bits / math.log2(q) if q > 1 else 0.0
    exact_capacity_q = float(width)
    return ApproxEqualizerSummary(
        q=int(q),
        width=int(width),
        threshold=float(threshold),
        threshold_mode=str(threshold_mode),
        alphabet_size_per_side=n_side,
        n_observed_pairs=observed_edges,
        exact_shared_classes=int(exact.shared_classes),
        exact_residual_qcoords=float(exact.residual_qcoords),
        exact_cost_qcoords=float(exact.cost_qcoords),
        exact_edge_density=float(exact.edge_density),
        approx_classes=int(approx_classes),
        approx_residual_qcoords=float(residual_q),
        approx_cost_qcoords=float(cost_q),
        approx_kept_edge_fraction=float(approx_n_edges / observed_edges) if observed_edges else 0.0,
        approx_kept_mass=float(kept_mass),
        approx_same_component_mass=float(same_mass),
        approx_cross_component_mass=float(cross_mass),
        approx_unassigned_mass=float(unassigned_mass),
        approx_component_mi_bits=float(mi_bits),
        approx_info_qcoords=float(info_q),
        approx_info_over_exact_capacity=float(info_q / exact_capacity_q) if exact_capacity_q > 0 else 0.0,
        approx_component_entropy_left_bits=float(h_left),
        approx_component_entropy_right_bits=float(h_right),
        approx_min_component_entropy_bits=float(min_h),
        approx_component_balance=float(balance),
        approx_class_sizes_left=tuple(int(x) for x in class_sizes_left),
        approx_class_sizes_right=tuple(int(x) for x in class_sizes_right),
    )


def approximate_shared_equalizer(
    left_words: np.ndarray,
    right_words: np.ndarray,
    q: int,
    width: int,
    threshold: float = 0.2,
    threshold_mode: ThresholdMode = "conditional",
) -> ApproxEqualizerSummary:
    M = compatibility_matrix(left_words, right_words, q, width)
    return approximate_shared_equalizer_from_matrix(M, q, width, threshold, threshold_mode)


# --------------------------------------------------------------------------- #
# Diamond measurement
# --------------------------------------------------------------------------- #
def _panel_word_matrix(ds: S.SCDCDiamond, maps: np.ndarray, states: np.ndarray) -> np.ndarray:
    left, right = IE._final_panel_words(ds, maps, states)  # reuse stable patch helper
    return compatibility_matrix(left, right, ds.q, ds.w)


def approx_equalizer_measure(
    ds: S.SCDCDiamond,
    thresholds: Iterable[float] = (0.2,),
    threshold_mode: ThresholdMode = "conditional",
    eq_mode: EqualizerMode = "joint",
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    precomputed_maps: tuple[np.ndarray, list[S.ScheduleInfo]] | None = None,
) -> list[dict]:
    """Measure exact and approximate equalizers on one diamond instance."""
    if eq_mode != "joint":
        # Pairwise approximate equalizers are less meaningful under noise because
        # each panel coordinate can split independently.  Keep the argument for
        # API compatibility, but intentionally gate it until there is a clear
        # use case.
        raise ValueError("approxequalizer currently supports eq_mode='joint' only")
    if precomputed_maps is None:
        maps, infos = S.step_maps(ds)
    else:
        maps, infos = precomputed_maps
    states = S._initial_states(ds, initial_pool, observer_init)
    M = _panel_word_matrix(ds, maps, states)
    rows = []
    for th in thresholds:
        summ = approximate_shared_equalizer_from_matrix(M, ds.q, ds.w, float(th), threshold_mode)
        out = summ.as_dict(prefix="aeq")
        out.update(
            eq_mode=eq_mode,
            kB=int(ds.kB),
            kC=int(ds.kC),
            kD=int(ds.kD),
            kM=int(ds.kC) if ds.kC == ds.kD else None,
            kA=int(ds.kA),
            k_total=int(ds.k_total),
            q=int(ds.q),
            w=int(ds.w),
            n_initial_states=int(len(states)),
            n_schedules=int(maps.shape[0]),
            initial_pool=initial_pool,
            observer_init=observer_init,
            extra_B=int(ds.meta["extra_B"]),
            extra_C=int(ds.meta["extra_C"]),
            extra_D=int(ds.meta["extra_D"]),
            extra_A=int(ds.meta["extra_A"]),
        )
        rows.append(out)
    return rows


# --------------------------------------------------------------------------- #
# Sweeps / analysis / plotting / CLI
# --------------------------------------------------------------------------- #
def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_modes(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def run_approx_equalizer_sweep(
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2, 3),
    ws: Iterable[int] = (1,),
    q: int = 3,
    n_instances: int = 20,
    rule_ensembles: Iterable[RuleEnsemble] = ("random", "copy", "permutive", "block", "canalizing"),
    mutation_rates: Iterable[float] = (0.0, 0.02, 0.05, 0.1, 0.2),
    thresholds: Iterable[float] = (0.15, 0.2, 0.25),
    threshold_mode: ThresholdMode = "conditional",
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    selection_threshold_qcoords: float = 1e-9,
    approx_info_threshold_qcoords: float = 0.2,
    min_same_component_mass: float = 0.6,
    n_blocks: int | None = None,
    mutate_all_rules: bool = True,
    extra_B: int | None = None,
    extra_M: int | None = None,
    extra_A: int | None = None,
    base_seed: int = 0,
    max_joint_states: int = 4 ** 9,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for kB in ks_B:
        for kM in ks_M:
            for kA in ks_A:
                if q ** (kB + 2 * kM + kA) > max_joint_states:
                    if verbose:
                        print(f"skip kB={kB}, kM={kM}, kA={kA}: joint state space too large", flush=True)
                    continue
                for w in ws:
                    if w > min(kB, kM) or kA < 2 * w:
                        continue
                    for ens in rule_ensembles:
                        for mu in mutation_rates:
                            for inst in range(n_instances):
                                seed = (
                                    base_seed * 150001
                                    + kB * 20011
                                    + kM * 5003
                                    + kA * 701
                                    + w * 37
                                    + int(round(float(mu) * 10000)) * 17
                                    + abs(hash(str(ens))) % 1009
                                    + inst
                                ) % 2**32
                                rng = np.random.default_rng(seed)
                                ds = S.make_scdc_diamond(
                                    kB=kB,
                                    kC=kM,
                                    kD=kM,
                                    kA=kA,
                                    q=q,
                                    w=w,
                                    rng=rng,
                                    extra_B=extra_B,
                                    extra_C=extra_M,
                                    extra_D=extra_M,
                                    extra_A=extra_A,
                                )
                                transforms = R.apply_rule_ensemble(
                                    ds,
                                    ens,  # type: ignore[arg-type]
                                    rng,
                                    mutation_rate=float(mu),
                                    n_blocks=n_blocks,
                                    mutate_all_rules=mutate_all_rules,
                                )
                                maps_infos = S.step_maps(ds)
                                measures = approx_equalizer_measure(
                                    ds,
                                    thresholds=thresholds,
                                    threshold_mode=threshold_mode,
                                    eq_mode="joint",
                                    initial_pool=initial_pool,
                                    observer_init=observer_init,
                                    precomputed_maps=maps_infos,
                                )
                                diag = R.interface_rule_diagnostics(ds)
                                for m in measures:
                                    m.update(diag)
                                    m.update(
                                        rule_ensemble=str(ens),
                                        mutation_rate=float(mu),
                                        threshold_mode=str(threshold_mode),
                                        selected_exact=bool(m["aeq_exact_residual_qcoords"] > selection_threshold_qcoords),
                                        selected_approx=bool(
                                            m["aeq_approx_info_qcoords"] >= approx_info_threshold_qcoords
                                            and m["aeq_approx_classes"] > 1
                                            and m["aeq_approx_same_component_mass"] >= min_same_component_mass
                                        ),
                                        approx_info_threshold_qcoords=float(approx_info_threshold_qcoords),
                                        min_same_component_mass=float(min_same_component_mass),
                                        seed=int(seed),
                                        instance=int(inst),
                                        transforms=json.dumps(transforms, default=str),
                                        n_blocks=int(R._default_n_blocks(q) if n_blocks is None else n_blocks),
                                    )
                                    rows.append(m)
                            if verbose:
                                print(
                                    f"approx-equalizer ensemble={ens}, mu={mu:g}, kB={kB}, kM={kM}, kA={kA}, w={w} done",
                                    flush=True,
                                )
    return pd.DataFrame(rows)


def analyze_approx_equalizer(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    eps = 1e-9
    bound_viol = 0
    bound_viol += int((df["aeq_approx_info_qcoords"] < -eps).sum())
    bound_viol += int((df["aeq_approx_info_qcoords"] > df["w"].astype(float) + eps).sum())
    bound_viol += int((df["aeq_approx_residual_qcoords"] < -eps).sum())
    bound_viol += int((df["aeq_approx_residual_qcoords"] > df["w"].astype(float) + eps).sum())

    group_cols = ["rule_ensemble", "mutation_rate", "aeq_threshold", "threshold_mode"]
    g = (
        df.groupby(group_cols)
        .agg(
            n=("selected_approx", "size"),
            exact_selection_fraction=("selected_exact", "mean"),
            approx_selection_fraction=("selected_approx", "mean"),
            exact_residual_qcoords=("aeq_exact_residual_qcoords", "mean"),
            approx_residual_qcoords=("aeq_approx_residual_qcoords", "mean"),
            approx_info_qcoords=("aeq_approx_info_qcoords", "mean"),
            approx_classes=("aeq_approx_classes", "mean"),
            same_component_mass=("aeq_approx_same_component_mass", "mean"),
            cross_component_mass=("aeq_approx_cross_component_mass", "mean"),
            kept_mass=("aeq_approx_kept_mass", "mean"),
            component_balance=("aeq_approx_component_balance", "mean"),
            interface_mi_norm=("rule_interface_mi_norm_mean", "mean"),
            interface_copy_accuracy=("rule_interface_copy_accuracy_mean", "mean"),
        )
        .reset_index()
    )
    by_ensemble = []
    for r in g.itertuples(index=False):
        by_ensemble.append(
            dict(
                rule_ensemble=str(r.rule_ensemble),
                mutation_rate=float(r.mutation_rate),
                threshold=float(r.aeq_threshold),
                threshold_mode=str(r.threshold_mode),
                n=int(r.n),
                exact_selection_fraction=float(r.exact_selection_fraction),
                approx_selection_fraction=float(r.approx_selection_fraction),
                exact_residual_qcoords=float(r.exact_residual_qcoords),
                approx_residual_qcoords=float(r.approx_residual_qcoords),
                approx_info_qcoords=float(r.approx_info_qcoords),
                approx_classes=float(r.approx_classes),
                same_component_mass=float(r.same_component_mass),
                cross_component_mass=float(r.cross_component_mass),
                kept_mass=float(r.kept_mass),
                component_balance=float(r.component_balance),
                interface_mi_norm=float(r.interface_mi_norm),
                interface_copy_accuracy=float(r.interface_copy_accuracy),
            )
        )

    # Main summary at the median/default threshold so mutation curves are easy
    # to read when multiple thresholds are swept.
    thresholds = sorted(float(x) for x in df["aeq_threshold"].unique())
    default_threshold = thresholds[len(thresholds) // 2]
    d0 = df[np.isclose(df["aeq_threshold"].astype(float), default_threshold)]
    zero = d0[np.isclose(d0["mutation_rate"].astype(float), 0.0)]
    random_zero = zero[zero["rule_ensemble"] == "random"]
    structured_zero = zero[zero["rule_ensemble"] != "random"]
    random_approx = float(random_zero["selected_approx"].mean()) if len(random_zero) else math.nan
    structured_approx = float(structured_zero["selected_approx"].mean()) if len(structured_zero) else math.nan
    random_exact = float(random_zero["selected_exact"].mean()) if len(random_zero) else math.nan
    structured_exact = float(structured_zero["selected_exact"].mean()) if len(structured_zero) else math.nan

    # Mutation curves at default threshold.
    mutation_curves = []
    for keys, sub in d0.groupby(["rule_ensemble", "threshold_mode"]):
        ens, mode = keys
        by_mu = sub.groupby("mutation_rate").agg(
            exact_residual=("aeq_exact_residual_qcoords", "mean"),
            approx_info=("aeq_approx_info_qcoords", "mean"),
            approx_residual=("aeq_approx_residual_qcoords", "mean"),
            approx_selection=("selected_approx", "mean"),
            same_mass=("aeq_approx_same_component_mass", "mean"),
            interface_mi=("rule_interface_mi_norm_mean", "mean"),
        )
        mutation_curves.append(
            dict(
                rule_ensemble=str(ens),
                threshold_mode=str(mode),
                threshold=float(default_threshold),
                exact_residual_by_mutation={float(k): float(v) for k, v in by_mu["exact_residual"].items()},
                approx_info_by_mutation={float(k): float(v) for k, v in by_mu["approx_info"].items()},
                approx_residual_by_mutation={float(k): float(v) for k, v in by_mu["approx_residual"].items()},
                approx_selection_by_mutation={float(k): float(v) for k, v in by_mu["approx_selection"].items()},
                same_mass_by_mutation={float(k): float(v) for k, v in by_mu["same_mass"].items()},
                interface_mi_by_mutation={float(k): float(v) for k, v in by_mu["interface_mi"].items()},
            )
        )

    # Does approximate recovery remain above exact recovery under mutation?
    mutated = d0[d0["mutation_rate"].astype(float) > 0]
    structured_mut = mutated[mutated["rule_ensemble"] != "random"]
    approx_minus_exact = float((structured_mut["aeq_approx_info_qcoords"] - structured_mut["aeq_exact_residual_qcoords"]).mean()) if len(structured_mut) else math.nan
    selected_mi = float(d0[d0["selected_approx"]]["rule_interface_mi_norm_mean"].mean()) if bool(d0["selected_approx"].any()) else math.nan
    unselected_mi = float(d0[~d0["selected_approx"]]["rule_interface_mi_norm_mean"].mean()) if bool((~d0["selected_approx"]).any()) else math.nan

    if bound_viol:
        verdict = "IMPLEMENTATION WARNING: approximate-equalizer bounds violated"
    elif not math.isnan(random_approx) and random_approx > 0.25:
        verdict = "APPROX THRESHOLD TOO WEAK: random rules often pass"
    elif not math.isnan(structured_approx) and structured_approx > 0.5 and (math.isnan(random_approx) or random_approx < 0.1):
        if not math.isnan(approx_minus_exact) and approx_minus_exact > 0.05:
            verdict = "APPROX-EQUALIZER SIGNAL: noisy structured rules retain latent interface labels beyond exact |S|"
        else:
            verdict = "APPROX-EQUALIZER SIGNAL: structured rules retain latent interface labels while random rules do not"
    elif not math.isnan(structured_approx) and structured_approx > 0.1 and (math.isnan(random_approx) or structured_approx > random_approx + 0.1):
        verdict = "PARTIAL APPROX-EQUALIZER SIGNAL: inspect thresholds and mutation curves"
    else:
        verdict = "NO ROBUST APPROX-EQUALIZER SIGNAL in this sweep"

    return dict(
        verdict=verdict,
        approximate_equalizer_bound_violations=int(bound_viol),
        default_threshold=float(default_threshold),
        random_exact_selection_at_zero_mutation=float(random_exact) if not math.isnan(random_exact) else None,
        structured_exact_selection_at_zero_mutation=float(structured_exact) if not math.isnan(structured_exact) else None,
        random_approx_selection_at_zero_mutation=float(random_approx) if not math.isnan(random_approx) else None,
        structured_approx_selection_at_zero_mutation=float(structured_approx) if not math.isnan(structured_approx) else None,
        mean_structured_mutation_approx_minus_exact=float(approx_minus_exact) if not math.isnan(approx_minus_exact) else None,
        mean_selected_interface_mi_norm=float(selected_mi) if not math.isnan(selected_mi) else None,
        mean_unselected_interface_mi_norm=float(unselected_mi) if not math.isnan(unselected_mi) else None,
        by_ensemble=by_ensemble,
        mutation_curves=mutation_curves,
    )


def plot_approx_equalizer(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    thresholds = sorted(float(x) for x in df["aeq_threshold"].unique())
    th = thresholds[len(thresholds) // 2]
    sub = df[np.isclose(df["aeq_threshold"].astype(float), th)]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for ens in sorted(sub.rule_ensemble.unique()):
        s = sub[sub.rule_ensemble == ens]
        g_exact = s.groupby("mutation_rate")["aeq_exact_residual_qcoords"].mean().dropna()
        g_approx = s.groupby("mutation_rate")["aeq_approx_info_qcoords"].mean().dropna()
        ax[0].plot(g_exact.index, g_exact.values, "o--", alpha=0.6, label=f"{ens} exact")
        ax[0].plot(g_approx.index, g_approx.values, "o-", label=f"{ens} approx")
    ax[0].set_xlabel("mutation rate")
    ax[0].set_ylabel(r"residual / information in $q$-coords")
    ax[0].set_title(f"exact vs approximate equalizer (threshold={th:g})")
    ax[0].legend(fontsize=6, ncol=2)

    for ens in sorted(sub.rule_ensemble.unique()):
        s = sub[sub.rule_ensemble == ens]
        g = s.groupby("mutation_rate")["selected_approx"].mean().dropna()
        ax[1].plot(g.index, g.values, "o-", label=str(ens))
    ax[1].set_xlabel("mutation rate")
    ax[1].set_ylabel("approx selection fraction")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("fraction with approximate latent S")
    ax[1].legend(fontsize=7)

    # Structure/residual scatter; random vs structured separation is often clear.
    ax[2].scatter(sub["rule_interface_mi_norm_mean"], sub["aeq_approx_info_qcoords"], s=18, alpha=0.7)
    ax[2].set_xlabel("mean interface-rule MI / log q")
    ax[2].set_ylabel(r"approx $I(Z_L;Z_R)/\log_2 q$")
    ax[2].set_title("latent information vs rule structure")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("q", type=int, nargs="?", default=3)
    ap.add_argument("--ks-B", default="2")
    ap.add_argument("--ks-M", default="2")
    ap.add_argument("--ks-A", default="2,3")
    ap.add_argument("--ws", default="1")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--rule-ensembles", default="random,copy,permutive,block,canalizing")
    ap.add_argument("--mutation-rates", default="0,0.02,0.05,0.1,0.2")
    ap.add_argument("--thresholds", default="0.15,0.2,0.25")
    ap.add_argument("--threshold-mode", choices=("conditional", "relative", "mass"), default="conditional")
    ap.add_argument("--initial-pool", choices=("all", "rec"), default="all")
    ap.add_argument("--observer-init", choices=("zero", "all"), default="zero")
    ap.add_argument("--selection-threshold", type=float, default=1e-9, help="exact equalizer residual threshold")
    ap.add_argument("--approx-info-threshold", type=float, default=0.2, help="approx info threshold in q-coordinates")
    ap.add_argument("--min-same-component-mass", type=float, default=0.6)
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--mutate-interface-only", action="store_true", help="mutate only declared interface rules instead of all rules")
    ap.add_argument("--extra-B", type=int, default=None)
    ap.add_argument("--extra-M", type=int, default=None)
    ap.add_argument("--extra-A", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--max-joint-states", type=int, default=4 ** 9)
    ap.add_argument("--out", default="example_results/approx_equalizer.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df = run_approx_equalizer_sweep(
        ks_B=_parse_ints(args.ks_B),
        ks_M=_parse_ints(args.ks_M),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        q=args.q,
        n_instances=args.instances,
        rule_ensembles=_parse_modes(args.rule_ensembles),  # type: ignore[arg-type]
        mutation_rates=_parse_floats(args.mutation_rates),
        thresholds=_parse_floats(args.thresholds),
        threshold_mode=args.threshold_mode,  # type: ignore[arg-type]
        initial_pool=args.initial_pool,  # type: ignore[arg-type]
        observer_init=args.observer_init,  # type: ignore[arg-type]
        selection_threshold_qcoords=args.selection_threshold,
        approx_info_threshold_qcoords=args.approx_info_threshold,
        min_same_component_mass=args.min_same_component_mass,
        n_blocks=args.n_blocks,
        mutate_all_rules=not args.mutate_interface_only,
        extra_B=args.extra_B,
        extra_M=args.extra_M,
        extra_A=args.extra_A,
        base_seed=args.base_seed,
        max_joint_states=args.max_joint_states,
        verbose=not args.quiet,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    res = analyze_approx_equalizer(df)
    print(json.dumps(res, indent=2, default=float))
    if args.plot:
        plot_approx_equalizer(df, args.plot)
        print(f"wrote {args.plot}")
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
