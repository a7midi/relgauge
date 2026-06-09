"""
commonfactoraudit.py -- audit live common factors in saved blind-selection winners.

The blind-selection experiment can now save the actual winning rule tables via
``--save-winners``.  This module loads those saved candidates and asks whether
the exact interface alphabet S is merely a formal compatibility partition or a
live common factor.

For each saved B -> {C,D} -> A diamond, the audit recomputes the exact joint
interface equalizer and then measures:

  * exact residual log_q |S| and shared class count;
  * entropy and balance of the selected label Z in the observed interface;
  * source dependence I(B_boundary ; Z);
  * sink readability I(Z ; A_left) and I(Z ; A_right);
  * optional one-step persistence I(Z_t ; Z_{t+1});
  * mutation survival of the exact residual.

Interpretation
--------------
A candidate is a strong finite law only if its shared alphabet is nontrivial and
live.  In this module "live" means that the label has entropy and depends on the
source boundary at a declared threshold.  The mutation curve then distinguishes
robust selected sectors from isolated truth-table accidents.

Winner files are local pickle artifacts.  Do not load winner files from
untrusted sources.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd

from . import core as C
from . import scdcdiamond as S
from . import interfaceequalizer as IE
from . import ruleselection as RS
from . import blindselection as BS

InitialPool = S.InitialPool
ObserverInit = S.ObserverInit


# --------------------------------------------------------------------------- #
# Parsing / small information helpers
# --------------------------------------------------------------------------- #
def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


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
    if x.shape != y.shape or len(x) == 0:
        return 0.0
    xv, xi = np.unique(x, return_inverse=True)
    yv, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xv), len(yv)), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)
    hx = _entropy_from_counts(joint.sum(axis=1))
    hy = _entropy_from_counts(joint.sum(axis=0))
    hxy = _entropy_from_counts(joint.reshape(-1))
    return float(hx + hy - hxy)


def _encode_word(vals: np.ndarray, q: int) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.int64)
    if vals.ndim == 1:
        return vals.copy()
    if vals.shape[1] == 0:
        return np.zeros(vals.shape[0], dtype=np.int64)
    powers = q ** np.arange(vals.shape[1], dtype=np.int64)
    return (vals * powers).sum(axis=1).astype(np.int64)


def _digit(indices: np.ndarray, q: int, vertex: int) -> np.ndarray:
    return (np.asarray(indices, dtype=np.int64) // (q ** int(vertex))) % q


def _word_digits(indices: np.ndarray, q: int, vertices: Iterable[int]) -> np.ndarray:
    verts = tuple(int(v) for v in vertices)
    if not verts:
        return np.zeros((len(indices), 0), dtype=np.int64)
    out = np.empty((len(indices), len(verts)), dtype=np.int64)
    for j, v in enumerate(verts):
        out[:, j] = _digit(indices, q, v)
    return out


class UF:
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


# --------------------------------------------------------------------------- #
# Exact common-factor reconstruction
# --------------------------------------------------------------------------- #
def source_words(ds: S.SCDCDiamond, states: np.ndarray) -> np.ndarray:
    """Initial B-boundary source word for every initial state."""
    verts = tuple(ds.offB + j for j in range(ds.w))
    return _encode_word(_word_digits(states, ds.q, verts), ds.q)




def scheduled_source_words(ds: S.SCDCDiamond, states: np.ndarray, infos: list[S.ScheduleInfo]) -> np.ndarray:
    """B-boundary source word after the B SCC has updated in each schedule.

    The interface pipeline reads B after the B part of the tick has run, not the
    raw previous-tick B value.  Using the post-B word is therefore the correct
    source variable for liveness of a selected B -> {C,D} -> A interface label.
    The output is flattened in the same schedule-major order as maps[:, states].
    """
    chunks = []
    verts = tuple(ds.offB + j for j in range(ds.w))
    cache: dict[tuple[int, ...], np.ndarray] = {}
    for info in infos:
        prefix = tuple(info.schedule[: ds.kB])
        if prefix not in cache:
            after_b = ds.joint.step_map(prefix)[states]
            cache[prefix] = _encode_word(_word_digits(after_b, ds.q, verts), ds.q)
        chunks.append(cache[prefix])
    return np.concatenate(chunks).astype(np.int64) if chunks else np.array([], dtype=np.int64)


def final_panel_words(ds: S.SCDCDiamond, finals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right A-panel words for final global states."""
    left_vertices = tuple(ds.offA + j for j in range(ds.w))
    right_vertices = tuple(ds.offA + ds.w + j for j in range(ds.w))
    left = _encode_word(_word_digits(finals, ds.q, left_vertices), ds.q)
    right = _encode_word(_word_digits(finals, ds.q, right_vertices), ds.q)
    return left, right


def exact_common_factor(left: np.ndarray, right: np.ndarray, q: int, width: int) -> dict:
    """Return exact equalizer labels for observed left/right word pairs.

    The exact common factor is the connected-component quotient of the observed
    bipartite compatibility graph on left and right interface words.  In
    addition to labels for the supplied observations, this function returns
    left/right label maps on the full side alphabet, with -1 for unseen words.
    Those maps are needed for persistence tests: the next tick must be read in
    the *same* selected alphabet S, rather than recomputing a new equalizer.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("left and right word arrays must have the same shape")
    n_side = int(q ** width)
    uf = UF(2 * n_side)
    pairs = np.unique(np.stack([left, right], axis=1), axis=0) if len(left) else np.zeros((0, 2), dtype=np.int64)
    touched_left: set[int] = set()
    touched_right: set[int] = set()
    for l, r in pairs:
        l_i = int(l); r_i = int(r)
        touched_left.add(l_i); touched_right.add(r_i)
        uf.union(l_i, n_side + r_i)
    roots = sorted({uf.find(x) for x in touched_left} | {uf.find(n_side + x) for x in touched_right})
    root_to_label = {root: i for i, root in enumerate(roots)}
    k = len(roots)
    left_map = np.full(n_side, -1, dtype=np.int64)
    right_map = np.full(n_side, -1, dtype=np.int64)
    for x in touched_left:
        left_map[int(x)] = int(root_to_label[uf.find(int(x))])
    for x in touched_right:
        right_map[int(x)] = int(root_to_label[uf.find(n_side + int(x))])
    if k == 0:
        zl = np.full(len(left), -1, dtype=np.int64)
        zr = np.full(len(right), -1, dtype=np.int64)
        residual = math.nan
    else:
        zl = left_map[left]
        zr = right_map[right]
        residual = math.log(k, q)
    left_sizes = []
    right_sizes = []
    for root in roots:
        left_sizes.append(sum(1 for x in touched_left if uf.find(x) == root))
        right_sizes.append(sum(1 for x in touched_right if uf.find(n_side + x) == root))
    return dict(
        z_left=zl,
        z_right=zr,
        left_label_map=left_map,
        right_label_map=right_map,
        shared_classes=int(k),
        residual_qcoords=float(residual),
        residual_bits=(float(math.log2(k)) if k > 0 else math.nan),
        observed_pairs=int(len(pairs)),
        edge_density=float(len(pairs) / (n_side * n_side)) if n_side else math.nan,
        left_class_sizes=tuple(int(x) for x in left_sizes),
        right_class_sizes=tuple(int(x) for x in right_sizes),
    )


def labels_in_existing_common_factor(left: np.ndarray, right: np.ndarray, cf: dict) -> tuple[np.ndarray, np.ndarray]:
    """Label panel pairs in an already selected alphabet S.

    Returns (labels, valid).  A pair is valid only when both side-words were
    present in the original equalizer and they map to the same S-label.
    This avoids the previous audit bug: recomputing the equalizer after adding
    next-tick pairs can merge labels and artificially produce NaN persistence.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    lmap = np.asarray(cf.get("left_label_map"), dtype=np.int64)
    rmap = np.asarray(cf.get("right_label_map"), dtype=np.int64)
    zl = np.full(len(left), -1, dtype=np.int64)
    zr = np.full(len(right), -2, dtype=np.int64)
    ok_l = (left >= 0) & (left < len(lmap))
    ok_r = (right >= 0) & (right < len(rmap))
    zl[ok_l] = lmap[left[ok_l]]
    zr[ok_r] = rmap[right[ok_r]]
    valid = (zl >= 0) & (zr >= 0) & (zl == zr)
    out = np.full(len(left), -1, dtype=np.int64)
    out[valid] = zl[valid]
    return out, valid



def _labels_from_map(words: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """Apply a fixed interface-word -> S-label map; unknown words map to -1."""
    words = np.asarray(words, dtype=np.int64)
    label_map = np.asarray(label_map, dtype=np.int64)
    out = np.full(len(words), -1, dtype=np.int64)
    ok = (words >= 0) & (words < len(label_map))
    out[ok] = label_map[words[ok]]
    return out


def _mutual_info_valid(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information after dropping pairs with unknown/negative labels."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    ok = (x >= 0) & (y >= 0)
    if not np.any(ok):
        return math.nan
    return _mutual_info_discrete(x[ok], y[ok])


def _entropy_valid(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.int64)
    ok = x >= 0
    if not np.any(ok):
        return math.nan
    return _entropy_from_counts(np.bincount(x[ok]))

def _subsample_indices(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=int(cap), replace=False).astype(np.int64))


# --------------------------------------------------------------------------- #
# One-candidate audit
# --------------------------------------------------------------------------- #
def audit_candidate(
    ds: S.SCDCDiamond,
    metrics: dict | None = None,
    *,
    candidate_index: int = 0,
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    live_threshold: float = 0.25,
    future_sample_cap: int = 200_000,
    mutation_rates: Iterable[float] = (0.0, 0.02, 0.05, 0.1),
    mutation_trials: int = 12,
    mutation_target: BS.TargetMode = "interface",
    mutation_entry_rate_scale: float = 1.0,
    base_seed: int = 0,
) -> dict:
    metrics = dict(metrics or {})
    maps, infos = S.step_maps(ds)
    states = S._initial_states(ds, initial_pool, observer_init)
    if len(states) == 0:
        raise ValueError("candidate has no initial states under requested pool/init")
    finals_grid = maps[:, states]
    finals = finals_grid.reshape(-1)
    left, right = final_panel_words(ds, finals)
    cf = exact_common_factor(left, right, ds.q, ds.w)
    z = cf["z_left"]
    zr = cf["z_right"]
    agreement = float(np.mean(z == zr)) if len(z) else math.nan

    src = scheduled_source_words(ds, states, infos)

    h_source = _entropy_from_counts(np.bincount(src, minlength=ds.q ** ds.w))
    h_z = _entropy_from_counts(np.bincount(z[z >= 0], minlength=max(1, int(cf["shared_classes"])))) if len(z) else 0.0
    h_left = _entropy_from_counts(np.bincount(left, minlength=ds.q ** ds.w))
    h_right = _entropy_from_counts(np.bincount(right, minlength=ds.q ** ds.w))
    mi_source_z = _mutual_info_discrete(src, z)
    mi_left_z = _mutual_info_discrete(left, z)
    mi_right_z = _mutual_info_discrete(right, z)
    mi_source_left = _mutual_info_discrete(src, left)
    mi_source_right = _mutual_info_discrete(src, right)
    max_bits = ds.w * math.log2(ds.q)

    # One-step persistence of the selected label.  The same exact S-label maps
    # selected at tick t are reused at tick t+1.  We deliberately do NOT
    # recompute the equalizer with next-tick pairs; doing so can merge old
    # components and falsely report NaN/zero persistence.
    persistence_mi = math.nan
    persistence_norm = math.nan
    persistence_valid_fraction = math.nan
    persistence_conservation_fraction = math.nan
    persistence_match_given_valid = math.nan
    persistence_valid_mi = math.nan
    persistence_valid_norm = math.nan
    if future_sample_cap != 0 and len(finals) > 0 and cf["shared_classes"] > 1:
        rng = np.random.default_rng((int(base_seed) * 1009 + int(candidate_index) * 9173 + 12345) % 2**32)
        eligible = np.flatnonzero(z >= 0)
        if len(eligible) > 0:
            idx0 = eligible[_subsample_indices(len(eligible), int(future_sample_cap), rng)]
            finals_sample = finals[idx0]
            z_now_sample = z[idx0]
            # Step each distinct first-tick final state forward by all admissible
            # second-tick schedules, then read the second-tick panels in the
            # original selected alphabet S.
            unique_finals, inv = np.unique(finals_sample, return_inverse=True)
            z_by_unique = np.zeros(len(unique_finals), dtype=np.int64)
            for u in range(len(unique_finals)):
                z_by_unique[u] = int(z_now_sample[np.flatnonzero(inv == u)[0]])
            next_states = maps[:, unique_finals].reshape(-1)
            z_now_rep = np.tile(z_by_unique, maps.shape[0])
            l2, r2 = final_panel_words(ds, next_states)
            z_next, valid_next = labels_in_existing_common_factor(l2, r2, cf)

            # Erasure-inclusive retention: invalid next labels are encoded as a
            # special erasure value.  This is conservative for conservation: an
            # erased label does not count as a conserved match, but it may still
            # carry information about which labels fail to persist.
            erasure_label = int(cf["shared_classes"])
            z_next_erasure = z_next.copy()
            z_next_erasure[~valid_next] = erasure_label
            persistence_mi = _mutual_info_discrete(z_now_rep, z_next_erasure)
            hz_now = _entropy_from_counts(np.bincount(z_now_rep[z_now_rep >= 0], minlength=int(cf["shared_classes"])))
            persistence_norm = float(persistence_mi / hz_now) if hz_now > 1e-12 else math.nan
            persistence_valid_fraction = float(np.mean(valid_next)) if len(valid_next) else math.nan
            persistence_conservation_fraction = float(np.mean(valid_next & (z_next == z_now_rep))) if len(valid_next) else math.nan
            if np.any(valid_next):
                persistence_match_given_valid = float(np.mean(z_next[valid_next] == z_now_rep[valid_next]))
                persistence_valid_mi = _mutual_info_discrete(z_now_rep[valid_next], z_next[valid_next])
                hz_valid = _entropy_from_counts(np.bincount(z_now_rep[valid_next], minlength=int(cf["shared_classes"])))
                persistence_valid_norm = float(persistence_valid_mi / hz_valid) if hz_valid > 1e-12 else math.nan

    # Mutation robustness: start from this exact winner and mutate only by a
    # declared blind mutation rate.  The audit uses exact residual, not the
    # approximate score.
    mut_rows = []
    for mu in mutation_rates:
        vals = []
        selected = []
        for t in range(int(mutation_trials)):
            rng = np.random.default_rng((int(base_seed) * 65537 + int(candidate_index) * 4099 + int(round(float(mu) * 10000)) * 17 + t) % 2**32)
            if float(mu) <= 0:
                child = BS.clone_diamond(ds)
            else:
                child, _ = BS.mutate_diamond(
                    ds,
                    rng,
                    target=mutation_target,
                    proposal_mode="entry",
                    entry_rate=float(mu) * float(mutation_entry_rate_scale),
                    table_rate=0.0,
                    ensure_change=True,
                )
            erow = IE.interface_equalizer_measure(child, eq_mode="joint", initial_pool=initial_pool, observer_init=observer_init, compare_product=False)
            r = float(erow.get("eq_residual_qcoords", 0.0))
            vals.append(r)
            selected.append(r > 1e-12)
        mut_rows.append(dict(
            mutation_rate=float(mu),
            mutation_exact_residual_mean=float(np.mean(vals)) if vals else math.nan,
            mutation_exact_residual_max=float(np.max(vals)) if vals else math.nan,
            mutation_selection_fraction=float(np.mean(selected)) if selected else math.nan,
        ))

    # Post-hoc local structure diagnostics.  These do not define the selected
    # law; they characterize the rule after selection.
    diag = RS.interface_rule_diagnostics(ds)

    source_norm = float(mi_source_z / max_bits) if max_bits > 1e-12 else math.nan
    source_over_hz = float(mi_source_z / h_z) if h_z > 1e-12 else math.nan
    balance = float(h_z / math.log2(cf["shared_classes"])) if cf["shared_classes"] > 1 else 0.0
    live = bool((cf["shared_classes"] > 1) and (source_over_hz >= float(live_threshold)))

    out = dict(
        candidate_index=int(candidate_index),
        q=int(ds.q),
        w=int(ds.w),
        kB=int(ds.kB),
        kC=int(ds.kC),
        kD=int(ds.kD),
        kA=int(ds.kA),
        exact_shared_classes=int(cf["shared_classes"]),
        exact_residual_qcoords=float(cf["residual_qcoords"]),
        exact_residual_bits=float(cf["residual_bits"]),
        exact_observed_pairs=int(cf["observed_pairs"]),
        exact_edge_density=float(cf["edge_density"]),
        exact_agreement_fraction=float(agreement),
        exact_left_class_sizes=json.dumps([int(x) for x in cf["left_class_sizes"]]),
        exact_right_class_sizes=json.dumps([int(x) for x in cf["right_class_sizes"]]),
        label_entropy_bits=float(h_z),
        label_balance=float(balance),
        source_entropy_bits=float(h_source),
        left_entropy_bits=float(h_left),
        right_entropy_bits=float(h_right),
        source_to_label_mi_bits=float(mi_source_z),
        source_to_label_mi_norm_capacity=float(source_norm),
        source_to_label_mi_over_label_entropy=float(source_over_hz),
        label_to_left_mi_bits=float(mi_left_z),
        label_to_right_mi_bits=float(mi_right_z),
        source_to_left_mi_bits=float(mi_source_left),
        source_to_right_mi_bits=float(mi_source_right),
        one_step_label_persistence_mi_bits=float(persistence_mi) if not math.isnan(persistence_mi) else math.nan,
        one_step_label_persistence_norm=float(persistence_norm) if not math.isnan(persistence_norm) else math.nan,
        one_step_label_valid_fraction=float(persistence_valid_fraction) if not math.isnan(persistence_valid_fraction) else math.nan,
        one_step_label_conservation_fraction=float(persistence_conservation_fraction) if not math.isnan(persistence_conservation_fraction) else math.nan,
        one_step_label_match_given_valid=float(persistence_match_given_valid) if not math.isnan(persistence_match_given_valid) else math.nan,
        one_step_label_valid_mi_bits=float(persistence_valid_mi) if not math.isnan(persistence_valid_mi) else math.nan,
        one_step_label_valid_norm=float(persistence_valid_norm) if not math.isnan(persistence_valid_norm) else math.nan,
        live_common_factor=bool(live),
        live_threshold=float(live_threshold),
        mutation_summary=json.dumps(mut_rows),
        mutation_survival_at_max_rate=float(mut_rows[-1]["mutation_selection_fraction"]) if mut_rows else math.nan,
        mutation_residual_at_max_rate=float(mut_rows[-1]["mutation_exact_residual_mean"]) if mut_rows else math.nan,
    )
    out.update({f"saved_{k}": v for k, v in dict(metrics).items() if k in (
        "search_mode", "run", "generation", "candidate", "blind_fitness",
        "blind_fitness_stage", "fitness_mode", "target_mode", "proposal_mode",
        "exact_residual_qcoords", "approx_quality_score", "approx_residual_qcoords",
    )})
    out.update(diag)
    return out


# --------------------------------------------------------------------------- #
# Bundle audit / analysis / plotting
# --------------------------------------------------------------------------- #
def load_winner_bundle(path: str) -> list[dict]:
    return BS.load_winner_records(path)


def run_common_factor_audit(
    winner_path: str,
    *,
    top_n: int = 20,
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    live_threshold: float = 0.25,
    future_sample_cap: int = 200_000,
    mutation_rates: Iterable[float] = (0.0, 0.02, 0.05, 0.1),
    mutation_trials: int = 12,
    mutation_target: BS.TargetMode = "interface",
    base_seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    records = load_winner_bundle(winner_path)
    if int(top_n) > 0:
        records = records[: int(top_n)]
    rows = []
    for i, rec in enumerate(records):
        ds = rec.get("diamond")
        if ds is None:
            raise ValueError(f"winner record {i} has no diamond object")
        m = rec.get("metrics", {})
        row = audit_candidate(
            ds,
            metrics=m,
            candidate_index=i,
            initial_pool=initial_pool,
            observer_init=observer_init,
            live_threshold=live_threshold,
            future_sample_cap=future_sample_cap,
            mutation_rates=mutation_rates,
            mutation_trials=mutation_trials,
            mutation_target=mutation_target,
            base_seed=base_seed,
        )
        rows.append(row)
        if verbose:
            print(
                f"audit winner={i} exact={row['exact_residual_qcoords']:.4f} "
                f"live={row['live_common_factor']} sourceMI/H={row['source_to_label_mi_over_label_entropy']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def analyze_common_factor_audit(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    live_frac = float(df["live_common_factor"].mean()) if "live_common_factor" in df else math.nan
    mean_exact = float(df["exact_residual_qcoords"].mean())
    mean_source = float(df["source_to_label_mi_over_label_entropy"].replace([np.inf, -np.inf], np.nan).mean())
    mean_persist = float(df["one_step_label_persistence_norm"].replace([np.inf, -np.inf], np.nan).mean()) if "one_step_label_persistence_norm" in df else math.nan
    mean_conserve = float(df["one_step_label_conservation_fraction"].replace([np.inf, -np.inf], np.nan).mean()) if "one_step_label_conservation_fraction" in df else math.nan
    mean_valid = float(df["one_step_label_valid_fraction"].replace([np.inf, -np.inf], np.nan).mean()) if "one_step_label_valid_fraction" in df else math.nan
    conserved_frac = float(((df.get("one_step_label_conservation_fraction", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan)) >= 0.75).mean()) if "one_step_label_conservation_fraction" in df else math.nan
    mean_mut = float(df["mutation_survival_at_max_rate"].replace([np.inf, -np.inf], np.nan).mean()) if "mutation_survival_at_max_rate" in df else math.nan
    full_frac = float((df["exact_residual_qcoords"] >= df["w"].astype(float) - 1e-9).mean())

    if live_frac >= 0.75 and mean_exact > 0.5 and (not math.isnan(mean_conserve)) and mean_conserve >= 0.75:
        verdict = "PERSISTENT LIVE COMMON-FACTOR SIGNAL: selected labels are source-live and dynamically conserved over one tick"
    elif live_frac >= 0.75 and mean_exact > 0.5:
        verdict = "LIVE COMMON-FACTOR SIGNAL: exact winner labels are source-dependent, but conservation/persistence is weak or unproven"
    elif mean_exact > 0.5 and live_frac < 0.25:
        verdict = "FORMAL EQUALIZER WARNING: exact shared alphabets exist but are weakly source-live"
    elif mean_exact > 0:
        verdict = "PARTIAL COMMON-FACTOR SIGNAL: nontrivial labels exist but liveness is mixed"
    else:
        verdict = "NO NONTRIVIAL COMMON FACTOR in saved winners"

    return dict(
        verdict=verdict,
        n_winners=int(len(df)),
        mean_exact_residual_qcoords=float(mean_exact),
        full_width_residual_fraction=float(full_frac),
        live_common_factor_fraction=float(live_frac),
        mean_source_to_label_mi_over_label_entropy=float(mean_source) if not math.isnan(mean_source) else None,
        mean_one_step_label_persistence_norm=float(mean_persist) if not math.isnan(mean_persist) else None,
        mean_one_step_label_valid_fraction=float(mean_valid) if not math.isnan(mean_valid) else None,
        mean_one_step_label_conservation_fraction=float(mean_conserve) if not math.isnan(mean_conserve) else None,
        conserved_common_factor_fraction=float(conserved_frac) if not math.isnan(conserved_frac) else None,
        mean_mutation_survival_at_max_rate=float(mean_mut) if not math.isnan(mean_mut) else None,
        top_winners=df.sort_values(["live_common_factor", "exact_residual_qcoords", "source_to_label_mi_over_label_entropy"], ascending=False)
        .head(10)[[
            "candidate_index", "exact_shared_classes", "exact_residual_qcoords",
            "live_common_factor", "source_to_label_mi_over_label_entropy",
            "label_balance", "one_step_label_persistence_norm",
            "one_step_label_valid_fraction", "one_step_label_conservation_fraction",
            "mutation_survival_at_max_rate",
        ]].to_dict(orient="records"),
    )


def plot_common_factor_audit(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(df))
    ax[0].bar(x, df["exact_residual_qcoords"].values)
    ax[0].set_xlabel("winner")
    ax[0].set_ylabel(r"exact residual $\log_q |S|$")
    ax[0].set_title("exact selected alphabet")

    ax[1].scatter(df["source_to_label_mi_over_label_entropy"], df["exact_residual_qcoords"], alpha=0.75)
    ax[1].set_xlabel(r"$I(source;S)/H(S)$")
    ax[1].set_ylabel(r"$\log_q |S|$")
    ax[1].set_title("liveness of selected label")

    # Mutation survival curves from JSON summaries.
    for _, row in df.head(8).iterrows():
        try:
            ms = json.loads(row["mutation_summary"])
        except Exception:
            continue
        ax[2].plot([m["mutation_rate"] for m in ms], [m["mutation_selection_fraction"] for m in ms], "o-", alpha=0.6)
    ax[2].set_xlabel("mutation rate")
    ax[2].set_ylabel("exact survival fraction")
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_title("winner mutation robustness")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("winner_path", help="pickle bundle from blindselection --save-winners")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--initial-pool", choices=("all", "rec"), default="all")
    ap.add_argument("--observer-init", choices=("zero", "all"), default="zero")
    ap.add_argument("--live-threshold", type=float, default=0.25)
    ap.add_argument("--future-sample-cap", type=int, default=200000)
    ap.add_argument("--mutation-rates", default="0,0.02,0.05,0.1")
    ap.add_argument("--mutation-trials", type=int, default=12)
    ap.add_argument("--mutation-target", choices=("all", "interface", "branch", "sink", "source"), default="interface")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", default="example_results/common_factor_audit.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df = run_common_factor_audit(
        args.winner_path,
        top_n=int(args.top_n),
        initial_pool=args.initial_pool,  # type: ignore[arg-type]
        observer_init=args.observer_init,  # type: ignore[arg-type]
        live_threshold=float(args.live_threshold),
        future_sample_cap=int(args.future_sample_cap),
        mutation_rates=_parse_floats(args.mutation_rates),
        mutation_trials=int(args.mutation_trials),
        mutation_target=args.mutation_target,  # type: ignore[arg-type]
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    res = analyze_common_factor_audit(df)
    print(json.dumps(res, indent=2, default=float))
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    if args.plot:
        plot_common_factor_audit(df, args.plot)
        print(f"wrote {args.plot}")
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
