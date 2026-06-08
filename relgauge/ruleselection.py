"""
ruleselection.py -- select local rule tables by nontrivial interface consistency.

Why this exists
---------------
The interface-equalizer experiment showed that maximum-entropy random rule
assignments usually give a trivial shared interface alphabet, |S|=1.  That is a
null-model result, not the end of the theory: random rules are a very specific
ensemble.  The selection question is:

    Which finite local update rules support a nontrivial shared interface
    alphabet S, and what structural features do those rules have?

This module compares rule ensembles on the same B -> {C,D} -> A diamond used by
``interfaceequalizer.py``.  It computes the equalizer residual

    residual = log_q |S|

for each rule assignment, marks selected instances with residual above a
threshold, and reports structural diagnostics of the rules that survive.

Implemented rule ensembles
--------------------------
``random``
    Maximum-entropy random rule tables.  This is the null model.

``copy``
    The interface pipeline copies a q-ary label from B through C/D into the two
    A panels.  This is the positive control: |S| should be q when all labels are
    reachable.

``permutive``
    Same as copy, but C and D apply independent permutations of the q symbols.
    The interface still has a q-component matching relation.

``block``
    The interface carries a coarse latent label with n_blocks <= q values.  For
    q=4 the default is two blocks, so residual is log_4(2)=0.5 in the ideal
    case.

``canalizing``
    A binary coarse interface: one special value is separated from all others.

``constant``
    The interface is deliberately collapsed to one symbol.  This is the
    negative structured control.

For structured ensembles, ``--mutation-rates`` can perturb every local rule
entry with a declared probability.  The resulting residual-vs-mutation curve is
a robustness diagnostic: selected rules are interesting only if they form a
basin rather than a single hand-tuned point.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

from . import core as C
from . import interfaceequalizer as IE
from . import scdcdiamond as S

RuleEnsemble = Literal["random", "copy", "permutive", "block", "canalizing", "constant"]
EqualizerMode = IE.EqualizerMode
InitialPool = S.InitialPool
ObserverInit = S.ObserverInit


# --------------------------------------------------------------------------- #
# Small information-theoretic helpers for rule diagnostics
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
    if x.shape != y.shape or len(x) == 0:
        return 0.0
    x_vals, xi = np.unique(x, return_inverse=True)
    y_vals, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(x_vals), len(y_vals)), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)
    hx = _entropy_from_counts(joint.sum(axis=1))
    hy = _entropy_from_counts(joint.sum(axis=0))
    hxy = _entropy_from_counts(joint.reshape(-1))
    return float(hx + hy - hxy)


def _all_inputs(indeg: int, q: int) -> np.ndarray:
    return C.all_states(int(indeg), int(q)) if indeg else np.zeros((1, 0), dtype=np.int64)


def _source_pos(preds: tuple[int, ...], src: int) -> int:
    try:
        return tuple(int(p) for p in preds).index(int(src))
    except ValueError as exc:
        raise ValueError(f"source {src} is not a predecessor of target with preds={preds}") from exc


def _rule_from_source(
    q: int,
    indeg: int,
    source_pos: int,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Return a rule table that depends only on one input coordinate."""
    X = _all_inputs(indeg, q)
    if transform is None:
        out = X[:, int(source_pos)]
    else:
        out = transform(X[:, int(source_pos)])
    return np.asarray(out, dtype=np.int64) % int(q)


def _constant_rule(q: int, indeg: int, value: int = 0) -> np.ndarray:
    return np.full(q ** int(indeg), int(value) % int(q), dtype=np.int64)


def _mutate_rule(rule: np.ndarray, q: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Independently mutate entries to a different q-symbol with probability rate."""
    out = np.asarray(rule, dtype=np.int64).copy()
    if rate <= 0 or out.size == 0:
        return out
    mask = rng.random(out.size) < float(rate)
    if not np.any(mask):
        return out
    jumps = rng.integers(1, q, size=int(mask.sum()), dtype=np.int64)
    out[mask] = (out[mask] + jumps) % int(q)
    return out


def _rule_interface_metrics(rule: np.ndarray, q: int, indeg: int, source_pos: int) -> dict:
    """How much a local rule preserves one declared input coordinate.

    We use a uniform distribution over input tuples.  The source input is the
    candidate interface label, and the output is the local update value.
    """
    X = _all_inputs(indeg, q)
    if X.shape[0] == 0:
        return dict(mi_bits=0.0, mi_norm=0.0, cond_entropy_bits=0.0, copy_accuracy=0.0, output_entropy_bits=0.0)
    src = X[:, int(source_pos)]
    y = np.asarray(rule, dtype=np.int64)
    mi = _mutual_info_discrete(src, y)
    h_src = math.log2(q)
    h_y = _entropy_from_counts(np.bincount(y, minlength=q))
    cond = max(0.0, h_y - mi)
    # Accuracy of the best deterministic decoder y=f(src), allowing arbitrary
    # relabeling / coarse maps rather than literal equality.
    correct = 0
    for s in range(q):
        ys = y[src == s]
        if len(ys):
            correct += int(np.bincount(ys, minlength=q).max())
    acc = correct / len(y) if len(y) else 0.0
    return dict(
        mi_bits=float(mi),
        mi_norm=float(mi / h_src) if h_src > 0 else 0.0,
        cond_entropy_bits=float(cond),
        copy_accuracy=float(acc),
        output_entropy_bits=float(h_y),
    )


# --------------------------------------------------------------------------- #
# Rule-ensemble construction
# --------------------------------------------------------------------------- #
def _default_n_blocks(q: int) -> int:
    if q <= 2:
        return 2
    return max(2, min(q, int(round(math.sqrt(q)))))


def _block_transform(q: int, n_blocks: int) -> Callable[[np.ndarray], np.ndarray]:
    n_blocks = max(1, min(int(n_blocks), int(q)))
    # Balanced-ish coarse label in [0,n_blocks).  The output alphabet remains q;
    # only the first n_blocks symbols are used by the interface.
    return lambda x: (np.asarray(x, dtype=np.int64) * n_blocks // int(q)).astype(np.int64)


def _canalizing_transform(q: int) -> Callable[[np.ndarray], np.ndarray]:
    # Separate value 0 from all nonzero values.
    return lambda x: (np.asarray(x, dtype=np.int64) != 0).astype(np.int64)


def _perm_transform(perm: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.asarray(perm, dtype=np.int64)[np.asarray(x, dtype=np.int64)]


def _set_rule_copying_source(ds: S.SCDCDiamond, target: int, source: int, transform: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
    preds = ds.joint.preds[int(target)]
    pos = _source_pos(preds, int(source))
    ds.joint.rules[int(target)] = _rule_from_source(ds.q, len(preds), pos, transform)


def _set_rule_constant(ds: S.SCDCDiamond, target: int, value: int = 0) -> None:
    ds.joint.rules[int(target)] = _constant_rule(ds.q, len(ds.joint.preds[int(target)]), value)


def _force_source_labels_in_B(ds: S.SCDCDiamond) -> None:
    """Make the B boundary vertices capable of carrying all q labels.

    With initial_pool='all', copying any local predecessor guarantees that the
    final B[j] can range over all q symbols.  This prevents false negatives in
    positive-control structured ensembles caused by a random B rule that happens
    to collapse the label before it reaches the interface.
    """
    # Stabilize the whole B SCC, not only the exported vertices.  Otherwise a
    # random predecessor of an exported vertex can collapse the label before the
    # copy chain begins.
    for v in range(ds.offB, ds.offC):
        preds = ds.joint.preds[v]
        if preds:
            ds.joint.rules[v] = _rule_from_source(ds.q, len(preds), 0, None)
        else:
            ds.joint.rules[v] = np.arange(ds.q, dtype=np.int64)[:1]


def apply_rule_ensemble(
    ds: S.SCDCDiamond,
    ensemble: RuleEnsemble,
    rng: np.random.Generator,
    mutation_rate: float = 0.0,
    n_blocks: int | None = None,
    mutate_all_rules: bool = True,
) -> dict:
    """Modify a random diamond in-place according to a rule ensemble."""
    q = int(ds.q)
    n_blocks = _default_n_blocks(q) if n_blocks is None else max(1, min(int(n_blocks), q))

    if ensemble == "random":
        # The diamond is already random.  Mutation would be statistically
        # indistinguishable from random resampling, so leave it as-is.
        transforms = dict(branch_C="random", branch_D="random", sink="random")
    else:
        _force_source_labels_in_B(ds)
        if ensemble == "copy":
            tC = tD = None
            transforms = dict(branch_C="identity", branch_D="identity", sink="identity")
        elif ensemble == "permutive":
            pC = rng.permutation(q).astype(np.int64)
            pD = rng.permutation(q).astype(np.int64)
            tC = _perm_transform(pC)
            tD = _perm_transform(pD)
            transforms = dict(branch_C=pC.tolist(), branch_D=pD.tolist(), sink="identity")
        elif ensemble == "block":
            tC = tD = _block_transform(q, n_blocks)
            transforms = dict(branch_C=f"block_{n_blocks}", branch_D=f"block_{n_blocks}", sink="identity")
        elif ensemble == "canalizing":
            tC = tD = _canalizing_transform(q)
            transforms = dict(branch_C="zero_vs_nonzero", branch_D="zero_vs_nonzero", sink="identity")
        elif ensemble == "constant":
            tC = tD = _block_transform(q, 1)
            transforms = dict(branch_C="constant", branch_D="constant", sink="constant")
        else:
            raise ValueError(f"unknown ensemble: {ensemble!r}")

        # B -> C/D interface: branch target stores a label derived from B[j].
        for j, ((b_src, c_tgt), (_, d_tgt)) in enumerate(zip(ds.interface_BC, ds.interface_BD)):
            if ensemble == "constant":
                _set_rule_constant(ds, c_tgt, 0)
                _set_rule_constant(ds, d_tgt, 0)
            else:
                _set_rule_copying_source(ds, c_tgt, b_src, tC)
                _set_rule_copying_source(ds, d_tgt, b_src, tD)

        # C/D -> A sink panels: A panels copy the branch label.  This makes the
        # equalizer probe the interface relation, not arbitrary random sink
        # mixing.  Hidden A vertices remain random unless mutation touches them.
        for (c_src, a_left), (d_src, a_right) in zip(ds.interface_CA, ds.interface_DA):
            if ensemble == "constant":
                _set_rule_constant(ds, a_left, 0)
                _set_rule_constant(ds, a_right, 0)
            else:
                _set_rule_copying_source(ds, a_left, c_src, None)
                _set_rule_copying_source(ds, a_right, d_src, None)

    if ensemble != "random" and mutation_rate > 0:
        targets = range(ds.k_total) if mutate_all_rules else _interface_rule_targets(ds)
        for v in targets:
            ds.joint.rules[int(v)] = _mutate_rule(ds.joint.rules[int(v)], ds.q, mutation_rate, rng)

    ds.joint.meta["rule_ensemble"] = str(ensemble)
    ds.joint.meta["mutation_rate"] = float(mutation_rate)
    ds.joint.meta["n_blocks"] = int(n_blocks)
    return transforms


def _interface_rule_targets(ds: S.SCDCDiamond) -> tuple[int, ...]:
    out: list[int] = []
    out.extend(ds.offB + j for j in range(ds.w))
    out.extend(t for _, t in ds.interface_BC)
    out.extend(t for _, t in ds.interface_BD)
    out.extend(t for _, t in ds.interface_CA)
    out.extend(t for _, t in ds.interface_DA)
    return tuple(sorted(set(int(x) for x in out)))


def interface_rule_diagnostics(ds: S.SCDCDiamond) -> dict:
    """Aggregate how strongly declared interface rules preserve their inputs."""
    metrics: list[dict] = []
    roles: list[str] = []

    # B source vertices: diagnose dependence on their first predecessor when it
    # exists.  This is mostly a label-availability control.
    for j in range(ds.w):
        v = ds.offB + j
        preds = ds.joint.preds[v]
        if preds:
            metrics.append(_rule_interface_metrics(ds.joint.rules[v], ds.q, len(preds), 0))
            roles.append("B_source")

    for b_src, c_tgt in ds.interface_BC:
        pos = _source_pos(ds.joint.preds[c_tgt], b_src)
        metrics.append(_rule_interface_metrics(ds.joint.rules[c_tgt], ds.q, len(ds.joint.preds[c_tgt]), pos))
        roles.append("B_to_C")
    for b_src, d_tgt in ds.interface_BD:
        pos = _source_pos(ds.joint.preds[d_tgt], b_src)
        metrics.append(_rule_interface_metrics(ds.joint.rules[d_tgt], ds.q, len(ds.joint.preds[d_tgt]), pos))
        roles.append("B_to_D")
    for c_src, a_left in ds.interface_CA:
        pos = _source_pos(ds.joint.preds[a_left], c_src)
        metrics.append(_rule_interface_metrics(ds.joint.rules[a_left], ds.q, len(ds.joint.preds[a_left]), pos))
        roles.append("C_to_A")
    for d_src, a_right in ds.interface_DA:
        pos = _source_pos(ds.joint.preds[a_right], d_src)
        metrics.append(_rule_interface_metrics(ds.joint.rules[a_right], ds.q, len(ds.joint.preds[a_right]), pos))
        roles.append("D_to_A")

    if not metrics:
        return {
            "rule_interface_mi_norm_mean": math.nan,
            "rule_interface_copy_accuracy_mean": math.nan,
            "rule_interface_cond_entropy_bits_mean": math.nan,
            "rule_interface_output_entropy_bits_mean": math.nan,
            "rule_interface_n": 0,
        }
    return {
        "rule_interface_mi_norm_mean": float(np.mean([m["mi_norm"] for m in metrics])),
        "rule_interface_mi_bits_mean": float(np.mean([m["mi_bits"] for m in metrics])),
        "rule_interface_copy_accuracy_mean": float(np.mean([m["copy_accuracy"] for m in metrics])),
        "rule_interface_cond_entropy_bits_mean": float(np.mean([m["cond_entropy_bits"] for m in metrics])),
        "rule_interface_output_entropy_bits_mean": float(np.mean([m["output_entropy_bits"] for m in metrics])),
        "rule_interface_n": int(len(metrics)),
        "rule_interface_roles": json.dumps(roles),
    }


# --------------------------------------------------------------------------- #
# Main measurement / sweeps
# --------------------------------------------------------------------------- #
def make_rule_selection_diamond(
    kB: int,
    kM: int,
    kA: int,
    q: int,
    w: int,
    rng: np.random.Generator,
    rule_ensemble: RuleEnsemble = "random",
    mutation_rate: float = 0.0,
    n_blocks: int | None = None,
    mutate_all_rules: bool = True,
    extra_B: int | None = None,
    extra_M: int | None = None,
    extra_A: int | None = None,
) -> S.SCDCDiamond:
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
    transforms = apply_rule_ensemble(
        ds,
        ensemble=rule_ensemble,
        rng=rng,
        mutation_rate=mutation_rate,
        n_blocks=n_blocks,
        mutate_all_rules=mutate_all_rules,
    )
    ds.meta.update(rule_ensemble=str(rule_ensemble), mutation_rate=float(mutation_rate), transforms=transforms)
    return ds


def rule_selection_measure(
    ds: S.SCDCDiamond,
    eq_mode: EqualizerMode = "joint",
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    selection_threshold_qcoords: float = 1e-9,
    compare_product: bool = False,
) -> dict:
    maps_infos = S.step_maps(ds)
    out = IE.interface_equalizer_measure(
        ds,
        eq_mode=eq_mode,
        initial_pool=initial_pool,
        observer_init=observer_init,
        compare_product=compare_product,
        precomputed_maps=maps_infos,
    )
    residual = float(out.get("eq_residual_qcoords", math.nan))
    out.update(interface_rule_diagnostics(ds))
    out.update(
        rule_ensemble=str(ds.meta.get("rule_ensemble", ds.joint.meta.get("rule_ensemble", "unknown"))),
        mutation_rate=float(ds.meta.get("mutation_rate", ds.joint.meta.get("mutation_rate", 0.0))),
        selected=bool(not math.isnan(residual) and residual > float(selection_threshold_qcoords)),
        selection_threshold_qcoords=float(selection_threshold_qcoords),
    )
    return out



def _stable_ensemble_code(name: str) -> int:
    # Deterministic across Python processes; do not use built-in hash().
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(str(name))) % 10007)

def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_modes(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def run_rule_selection_sweep(
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2, 3),
    ws: Iterable[int] = (1,),
    q: int = 4,
    n_instances: int = 20,
    rule_ensembles: Iterable[RuleEnsemble] = ("random", "copy", "permutive", "block", "constant"),
    mutation_rates: Iterable[float] = (0.0,),
    eq_modes: Iterable[EqualizerMode] = ("joint",),
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    selection_threshold_qcoords: float = 1e-9,
    n_blocks: int | None = None,
    mutate_all_rules: bool = True,
    compare_product: bool = False,
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
                                    base_seed * 1000037
                                    + kB * 524287
                                    + kM * 8191
                                    + kA * 257
                                    + w * 31
                                    + {"random": 0, "copy": 1, "permutive": 2, "block": 3, "canalizing": 4, "constant": 5}[str(ens)] * 10007
                                    + int(round(float(mu) * 10000)) * 17
                                    + inst
                                ) % 2**32
                                rng = np.random.default_rng(seed)
                                ds = make_rule_selection_diamond(
                                    kB=kB,
                                    kM=kM,
                                    kA=kA,
                                    q=q,
                                    w=w,
                                    rng=rng,
                                    rule_ensemble=ens,  # type: ignore[arg-type]
                                    mutation_rate=float(mu),
                                    n_blocks=n_blocks,
                                    mutate_all_rules=mutate_all_rules,
                                    extra_B=extra_B,
                                    extra_M=extra_M,
                                    extra_A=extra_A,
                                )
                                for eq_mode in eq_modes:
                                    m = rule_selection_measure(
                                        ds,
                                        eq_mode=eq_mode,  # type: ignore[arg-type]
                                        initial_pool=initial_pool,
                                        observer_init=observer_init,
                                        selection_threshold_qcoords=selection_threshold_qcoords,
                                        compare_product=compare_product,
                                    )
                                    m.update(seed=int(seed), instance=int(inst), n_blocks=int(_default_n_blocks(q) if n_blocks is None else n_blocks))
                                    rows.append(m)
                            if verbose:
                                print(
                                    f"rule-selection ensemble={ens}, mu={mu:g}, kB={kB}, kM={kM}, kA={kA}, w={w} done",
                                    flush=True,
                                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Analysis / plotting / CLI
# --------------------------------------------------------------------------- #
def analyze_rule_selection(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    eps = 1e-9
    bound_viol = int((df["eq_residual_qcoords"] < -eps).sum())
    bound_viol += int((df["eq_residual_qcoords"] > df["w"].astype(float) + eps).sum())
    bound_viol += int((df["eq_cost_qcoords"] < -eps).sum())
    bound_viol += int((df["eq_cost_qcoords"] > 2 * df["w"].astype(float) + eps).sum())

    group_cols = ["rule_ensemble", "mutation_rate", "eq_mode"]
    g = (
        df.groupby(group_cols)
        .agg(
            n=("selected", "size"),
            selection_fraction=("selected", "mean"),
            residual_qcoords=("eq_residual_qcoords", "mean"),
            residual_per_width=("eq_residual_per_width", "mean"),
            cost_qcoords=("eq_cost_qcoords", "mean"),
            shared_classes=("eq_shared_classes_mean", "mean"),
            edge_density=("eq_edge_density_mean", "mean"),
            interface_mi_norm=("rule_interface_mi_norm_mean", "mean"),
            interface_copy_accuracy=("rule_interface_copy_accuracy_mean", "mean"),
            interface_cond_entropy_bits=("rule_interface_cond_entropy_bits_mean", "mean"),
        )
        .reset_index()
    )
    by_ensemble = []
    for r in g.itertuples(index=False):
        by_ensemble.append(
            dict(
                rule_ensemble=str(r.rule_ensemble),
                mutation_rate=float(r.mutation_rate),
                eq_mode=str(r.eq_mode),
                n=int(r.n),
                selection_fraction=float(r.selection_fraction),
                residual_qcoords=float(r.residual_qcoords),
                residual_per_width=float(r.residual_per_width),
                cost_qcoords=float(r.cost_qcoords),
                shared_classes=float(r.shared_classes),
                edge_density=float(r.edge_density),
                interface_mi_norm=float(r.interface_mi_norm),
                interface_copy_accuracy=float(r.interface_copy_accuracy),
                interface_cond_entropy_bits=float(r.interface_cond_entropy_bits),
            )
        )

    # Mutation stability: for each ensemble/mode, report selected fraction and
    # residual as a function of mutation rate.
    mutation_curves = []
    for keys, sub in df.groupby(["rule_ensemble", "eq_mode"]):
        ens, mode = keys
        by_mu = sub.groupby("mutation_rate").agg(
            selection_fraction=("selected", "mean"),
            residual_qcoords=("eq_residual_qcoords", "mean"),
            interface_mi_norm=("rule_interface_mi_norm_mean", "mean"),
        )
        mutation_curves.append(
            dict(
                rule_ensemble=str(ens),
                eq_mode=str(mode),
                selection_fraction_by_mutation={float(k): float(v) for k, v in by_mu["selection_fraction"].items()},
                residual_by_mutation={float(k): float(v) for k, v in by_mu["residual_qcoords"].items()},
                interface_mi_by_mutation={float(k): float(v) for k, v in by_mu["interface_mi_norm"].items()},
            )
        )

    zero = df[np.isclose(df["mutation_rate"].astype(float), 0.0)]
    random_zero = zero[zero["rule_ensemble"] == "random"]
    structured_zero = zero[zero["rule_ensemble"] != "random"]
    random_sel = float(random_zero["selected"].mean()) if len(random_zero) else math.nan
    structured_sel = float(structured_zero["selected"].mean()) if len(structured_zero) else math.nan
    best_structured = (
        structured_zero.groupby("rule_ensemble")["selected"].mean().max()
        if len(structured_zero)
        else math.nan
    )
    mean_selected_mi = float(df[df["selected"]]["rule_interface_mi_norm_mean"].mean()) if bool(df["selected"].any()) else math.nan
    mean_unselected_mi = float(df[~df["selected"]]["rule_interface_mi_norm_mean"].mean()) if bool((~df["selected"]).any()) else math.nan

    if bound_viol:
        verdict = "IMPLEMENTATION WARNING: equalizer bounds violated"
    elif not math.isnan(random_sel) and random_sel > 0.25:
        verdict = "SELECTION CRITERION TOO WEAK: random rules often pass"
    elif not math.isnan(best_structured) and best_structured > 0.5 and (math.isnan(random_sel) or random_sel < 0.1):
        verdict = "RULE-SELECTION SIGNAL: nonrandom rule structure supports nontrivial interface equalizers while random rules do not"
    elif not math.isnan(structured_sel) and structured_sel > 0.1 and (math.isnan(random_sel) or structured_sel > random_sel + 0.1):
        verdict = "PARTIAL RULE-SELECTION SIGNAL: selected rules exist but robustness/structure needs inspection"
    else:
        verdict = "NO ROBUST RULE-SELECTION SIGNAL in this sweep"

    return dict(
        verdict=verdict,
        equalizer_bound_violations=int(bound_viol),
        random_selection_fraction_at_zero_mutation=float(random_sel) if not math.isnan(random_sel) else None,
        structured_selection_fraction_at_zero_mutation=float(structured_sel) if not math.isnan(structured_sel) else None,
        best_structured_selection_fraction_at_zero_mutation=float(best_structured) if not math.isnan(best_structured) else None,
        mean_selected_interface_mi_norm=float(mean_selected_mi) if not math.isnan(mean_selected_mi) else None,
        mean_unselected_interface_mi_norm=float(mean_unselected_mi) if not math.isnan(mean_unselected_mi) else None,
        by_ensemble=by_ensemble,
        mutation_curves=mutation_curves,
    )


def plot_rule_selection(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # Residual vs mutation curves, one line per ensemble (averaged over eq_mode).
    for ens in sorted(df.rule_ensemble.unique()):
        sub = df[df.rule_ensemble == ens]
        g = sub.groupby("mutation_rate")["eq_residual_qcoords"].mean().dropna()
        ax[0].plot(g.index, g.values, "o-", label=str(ens))
    ax[0].set_xlabel("mutation rate")
    ax[0].set_ylabel(r"equalizer residual $\log_q |S|$")
    ax[0].set_title("selected interface alphabet")
    ax[0].legend(fontsize=7)

    for ens in sorted(df.rule_ensemble.unique()):
        sub = df[df.rule_ensemble == ens]
        g = sub.groupby("mutation_rate")["selected"].mean().dropna()
        ax[1].plot(g.index, g.values, "o-", label=str(ens))
    ax[1].set_xlabel("mutation rate")
    ax[1].set_ylabel("selection fraction")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("fraction with nontrivial S")
    ax[1].legend(fontsize=7)

    ax[2].scatter(df["rule_interface_mi_norm_mean"], df["eq_residual_qcoords"], s=18, alpha=0.7)
    ax[2].set_xlabel("mean interface-rule MI / log q")
    ax[2].set_ylabel(r"equalizer residual $\log_q |S|$")
    ax[2].set_title("structure vs selected residual")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("q", type=int, nargs="?", default=4)
    ap.add_argument("--ks-B", default="2")
    ap.add_argument("--ks-M", default="2")
    ap.add_argument("--ks-A", default="2,3")
    ap.add_argument("--ws", default="1")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--rule-ensembles", default="random,copy,permutive,block,canalizing,constant")
    ap.add_argument("--mutation-rates", default="0")
    ap.add_argument("--eq-modes", default="joint")
    ap.add_argument("--initial-pool", choices=("all", "rec"), default="all")
    ap.add_argument("--observer-init", choices=("zero", "all"), default="zero")
    ap.add_argument("--selection-threshold", type=float, default=1e-9)
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--mutate-interface-only", action="store_true", help="mutate only declared interface rules instead of all rules")
    ap.add_argument("--compare-product", action="store_true", help="also compute old product-local sink_equal cost")
    ap.add_argument("--extra-B", type=int, default=None)
    ap.add_argument("--extra-M", type=int, default=None)
    ap.add_argument("--extra-A", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--max-joint-states", type=int, default=4 ** 9)
    ap.add_argument("--out", default="example_results/rule_selection.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df = run_rule_selection_sweep(
        ks_B=_parse_ints(args.ks_B),
        ks_M=_parse_ints(args.ks_M),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        q=args.q,
        n_instances=args.instances,
        rule_ensembles=_parse_modes(args.rule_ensembles),  # type: ignore[arg-type]
        mutation_rates=_parse_floats(args.mutation_rates),
        eq_modes=_parse_modes(args.eq_modes),  # type: ignore[arg-type]
        initial_pool=args.initial_pool,  # type: ignore[arg-type]
        observer_init=args.observer_init,  # type: ignore[arg-type]
        selection_threshold_qcoords=args.selection_threshold,
        n_blocks=args.n_blocks,
        mutate_all_rules=not args.mutate_interface_only,
        compare_product=args.compare_product,
        extra_B=args.extra_B,
        extra_M=args.extra_M,
        extra_A=args.extra_A,
        base_seed=args.base_seed,
        max_joint_states=args.max_joint_states,
        verbose=not args.quiet,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    res = analyze_rule_selection(df)
    print(json.dumps(res, indent=2, default=float))
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    if args.plot:
        plot_rule_selection(df, args.plot)
        print(f"wrote {args.plot}")
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
