"""
temporalchain.py -- two-diamond temporal-chain transport/composition test.

Why this exists
---------------
A single diamond

        B
       / \
      C   D
       \ /
        A

is a convergence topology.  It can select an exact shared alphabet S at the
sink, but that alone is a consistency constraint, not a conservation law.
Conservation requires transport: a selected label should survive through a
second convergence.

This module builds the minimal two-diamond chain

        B0
       /  \
     C1    D1
       \  /
        A1
       /  \
     C2    D2
       \  /
        A2

where A1's two panels feed the second diamond.  We measure:

  S1: exact equalizer label at A1 after the first convergence
  S2: exact equalizer label at A2 after the second convergence

and report whether S1 transports to S2:

  I(S1;S2)/H(S2), best deterministic S1->S2 accuracy,
  and source-to-final-label information I(B_sent;S2)/H(S2).

The default initial ensemble varies only B0 and sets all downstream components
to zero.  This is intentional: it tests causal transport of a source label
through the chain without conflating it with arbitrary downstream initial
memory.
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
from . import ruleselection as R

RuleEnsemble = Literal["random", "copy", "permutive", "block", "canalizing", "constant"]
InitialMode = Literal["source_all", "source_random", "joint_random"]


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


def _entropy_discrete(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    _, counts = np.unique(np.asarray(x, dtype=np.int64), return_counts=True)
    return _entropy_from_counts(counts)


def _mutual_info_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if x.shape != y.shape or len(x) == 0:
        return 0.0
    xv, xi = np.unique(x, return_inverse=True)
    yv, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xv), len(yv)), dtype=np.int64)
    np.add.at(joint, (xi, yi), 1)
    return float(
        _entropy_from_counts(joint.sum(axis=1))
        + _entropy_from_counts(joint.sum(axis=0))
        - _entropy_from_counts(joint.reshape(-1))
    )


def _best_map_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    """Best deterministic y=f(x) accuracy on observed samples."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if x.shape != y.shape or len(x) == 0:
        return 0.0
    correct = 0
    for a in np.unique(x):
        ys = y[x == a]
        correct += int(np.bincount(ys).max()) if len(ys) else 0
    return float(correct / len(x))


def _variation_of_information_norm(x: np.ndarray, y: np.ndarray) -> float:
    """A symmetric 0..1-ish dissimilarity; 0 means identical up to relabeling."""
    hx = _entropy_discrete(x)
    hy = _entropy_discrete(y)
    mi = _mutual_info_discrete(x, y)
    den = max(hx, hy, 1e-12)
    return float(max(0.0, (hx + hy - 2 * mi) / den))


# --------------------------------------------------------------------------- #
# Small state helpers
# --------------------------------------------------------------------------- #
def _decode_indices(indices: np.ndarray, k: int, q: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    S = np.empty((len(indices), k), dtype=np.int64)
    for v in range(k):
        S[:, v] = (indices // (q ** v)) % q
    return S


def _encode_states(S: np.ndarray, q: int) -> np.ndarray:
    powers = q ** np.arange(S.shape[1], dtype=np.int64)
    return (S.astype(np.int64) * powers).sum(axis=1).astype(np.int64)


def _step_subset(sys: C.RelationalSystem, states: np.ndarray, schedule: tuple[int, ...]) -> np.ndarray:
    """Apply one sequential in-place schedule to a subset of states only."""
    S = _decode_indices(states, sys.k, sys.q)
    for v in schedule:
        preds = sys.preds[int(v)]
        if not preds:
            continue
        vals = S[:, list(preds)]
        powers = sys.q ** np.arange(len(preds), dtype=np.int64)
        idx = (vals * powers).sum(axis=1).astype(np.int64)
        S[:, int(v)] = sys.rules[int(v)][idx]
    return _encode_states(S, sys.q)


def _word(indices: np.ndarray, q: int, vertices: Iterable[int]) -> np.ndarray:
    verts = tuple(int(v) for v in vertices)
    if not verts:
        return np.zeros(len(indices), dtype=np.int64)
    out = np.zeros(len(indices), dtype=np.int64)
    for j, v in enumerate(verts):
        out += ((np.asarray(indices, dtype=np.int64) // (q ** v)) % q) * (q ** j)
    return out.astype(np.int64)


# --------------------------------------------------------------------------- #
# Exact equalizer labels
# --------------------------------------------------------------------------- #
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

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra != rb:
            self.p[ra] = rb


def exact_equalizer_labels(left_words: np.ndarray, right_words: np.ndarray, q: int, w: int) -> dict:
    """Return exact equalizer labels for observed left/right word pairs."""
    left_words = np.asarray(left_words, dtype=np.int64)
    right_words = np.asarray(right_words, dtype=np.int64)
    if left_words.shape != right_words.shape:
        raise ValueError("left and right must have same shape")
    n_side = int(q ** w)
    if len(left_words) == 0:
        return dict(shared_classes=0, residual_qcoords=math.nan, labels=np.array([], dtype=np.int64), edge_density=math.nan)
    uf = _UF(2 * n_side)
    pairs = np.unique(np.stack([left_words, right_words], axis=1), axis=0)
    for l, r in pairs:
        uf.union(int(l), n_side + int(r))
    roots = sorted({uf.find(int(l)) for l in left_words} | {uf.find(n_side + int(r)) for r in right_words})
    root_to_label = {root: i for i, root in enumerate(roots)}
    labels = np.array([root_to_label[uf.find(int(l))] for l in left_words], dtype=np.int64)
    shared = len(roots)
    return dict(
        shared_classes=int(shared),
        residual_qcoords=float(math.log(shared, q)) if shared > 0 else math.nan,
        labels=labels,
        edge_density=float(len(pairs) / (n_side * n_side)),
        pairs=np.asarray(pairs, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Chain construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TemporalChain:
    joint: C.RelationalSystem
    kB: int
    kM: int
    kA: int
    q: int
    w: int
    offB: int
    offC1: int
    offD1: int
    offA1: int
    offC2: int
    offD2: int
    offA2: int
    interface_BC1: tuple[tuple[int, int], ...]
    interface_BD1: tuple[tuple[int, int], ...]
    interface_C1A1: tuple[tuple[int, int], ...]
    interface_D1A1: tuple[tuple[int, int], ...]
    interface_A1C2: tuple[tuple[int, int], ...]
    interface_A1D2: tuple[tuple[int, int], ...]
    interface_C2A2: tuple[tuple[int, int], ...]
    interface_D2A2: tuple[tuple[int, int], ...]
    meta: dict

    @property
    def k_total(self) -> int:
        return self.joint.k

    @property
    def a1_left(self) -> tuple[int, ...]:
        return tuple(self.offA1 + j for j in range(self.w))

    @property
    def a1_right(self) -> tuple[int, ...]:
        return tuple(self.offA1 + self.w + j for j in range(self.w))

    @property
    def a2_left(self) -> tuple[int, ...]:
        return tuple(self.offA2 + j for j in range(self.w))

    @property
    def a2_right(self) -> tuple[int, ...]:
        return tuple(self.offA2 + self.w + j for j in range(self.w))

    @property
    def b_boundary(self) -> tuple[int, ...]:
        return tuple(self.offB + j for j in range(self.w))


def _add_component(pred_sets: list[set[int]], off: int, k: int, q: int, rng: np.random.Generator, extra: int | None) -> None:
    comp = C.make_random_scc(k, q, k if extra is None else int(extra), rng)
    for v in range(k):
        pred_sets[off + v].update(off + p for p in comp.preds[v])


def _random_rule(q: int, indeg: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, q, size=q ** int(indeg), dtype=np.int64)


def make_temporal_chain(
    kB: int,
    kM: int,
    kA: int,
    q: int,
    w: int,
    rng: np.random.Generator,
    extra_B: int | None = None,
    extra_M: int | None = None,
    extra_A: int | None = None,
) -> TemporalChain:
    if w <= 0:
        raise ValueError("w must be positive")
    if w > min(kB, kM):
        raise ValueError("w must be <= min(kB,kM)")
    if kA < 2 * w:
        raise ValueError("kA must be at least 2*w")
    offB = 0
    offC1 = offB + kB
    offD1 = offC1 + kM
    offA1 = offD1 + kM
    offC2 = offA1 + kA
    offD2 = offC2 + kM
    offA2 = offD2 + kM
    k_total = offA2 + kA
    pred_sets: list[set[int]] = [set() for _ in range(k_total)]
    _add_component(pred_sets, offB, kB, q, rng, extra_B)
    _add_component(pred_sets, offC1, kM, q, rng, extra_M)
    _add_component(pred_sets, offD1, kM, q, rng, extra_M)
    _add_component(pred_sets, offA1, kA, q, rng, extra_A)
    _add_component(pred_sets, offC2, kM, q, rng, extra_M)
    _add_component(pred_sets, offD2, kM, q, rng, extra_M)
    _add_component(pred_sets, offA2, kA, q, rng, extra_A)

    interface_BC1: list[tuple[int, int]] = []
    interface_BD1: list[tuple[int, int]] = []
    interface_C1A1: list[tuple[int, int]] = []
    interface_D1A1: list[tuple[int, int]] = []
    interface_A1C2: list[tuple[int, int]] = []
    interface_A1D2: list[tuple[int, int]] = []
    interface_C2A2: list[tuple[int, int]] = []
    interface_D2A2: list[tuple[int, int]] = []

    for j in range(w):
        b = offB + j
        c1 = offC1 + j
        d1 = offD1 + j
        pred_sets[c1].add(b)
        pred_sets[d1].add(b)
        interface_BC1.append((b, c1))
        interface_BD1.append((b, d1))

        c1s = offC1 + j
        d1s = offD1 + j
        a1l = offA1 + j
        a1r = offA1 + w + j
        pred_sets[a1l].add(c1s)
        pred_sets[a1r].add(d1s)
        interface_C1A1.append((c1s, a1l))
        interface_D1A1.append((d1s, a1r))

        c2 = offC2 + j
        d2 = offD2 + j
        pred_sets[c2].add(a1l)
        pred_sets[d2].add(a1r)
        interface_A1C2.append((a1l, c2))
        interface_A1D2.append((a1r, d2))

        a2l = offA2 + j
        a2r = offA2 + w + j
        pred_sets[a2l].add(offC2 + j)
        pred_sets[a2r].add(offD2 + j)
        interface_C2A2.append((offC2 + j, a2l))
        interface_D2A2.append((offD2 + j, a2r))

    preds = [tuple(sorted(x)) for x in pred_sets]
    rules = [_random_rule(q, len(preds[v]), rng) for v in range(k_total)]
    joint = C.RelationalSystem(k_total, q, preds, rules, meta=dict(ensemble="temporal_chain", kB=kB, kM=kM, kA=kA, w=w))
    return TemporalChain(
        joint=joint,
        kB=int(kB),
        kM=int(kM),
        kA=int(kA),
        q=int(q),
        w=int(w),
        offB=offB,
        offC1=offC1,
        offD1=offD1,
        offA1=offA1,
        offC2=offC2,
        offD2=offD2,
        offA2=offA2,
        interface_BC1=tuple(interface_BC1),
        interface_BD1=tuple(interface_BD1),
        interface_C1A1=tuple(interface_C1A1),
        interface_D1A1=tuple(interface_D1A1),
        interface_A1C2=tuple(interface_A1C2),
        interface_A1D2=tuple(interface_A1D2),
        interface_C2A2=tuple(interface_C2A2),
        interface_D2A2=tuple(interface_D2A2),
        meta=dict(joint.meta),
    )


# --------------------------------------------------------------------------- #
# Rule ensembles
# --------------------------------------------------------------------------- #
def _source_pos(preds: tuple[int, ...], src: int) -> int:
    try:
        return tuple(int(x) for x in preds).index(int(src))
    except ValueError as exc:
        raise ValueError(f"source {src} not in preds {preds}") from exc


def _all_inputs(indeg: int, q: int) -> np.ndarray:
    return C.all_states(int(indeg), int(q)) if indeg else np.zeros((1, 0), dtype=np.int64)


def _rule_from_source(q: int, indeg: int, source_pos: int, transform: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    X = _all_inputs(indeg, q)
    src = X[:, int(source_pos)]
    if transform is not None:
        src = transform(src)
    return np.asarray(src, dtype=np.int64) % int(q)


def _constant_rule(q: int, indeg: int, value: int = 0) -> np.ndarray:
    return np.full(q ** int(indeg), int(value) % int(q), dtype=np.int64)


def _set_copy(tc: TemporalChain, target: int, source: int, transform: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
    preds = tc.joint.preds[int(target)]
    pos = _source_pos(preds, int(source))
    tc.joint.rules[int(target)] = _rule_from_source(tc.q, len(preds), pos, transform)


def _set_constant(tc: TemporalChain, target: int, value: int = 0) -> None:
    tc.joint.rules[int(target)] = _constant_rule(tc.q, len(tc.joint.preds[int(target)]), value)


def _mutate_rule(rule: np.ndarray, q: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(rule, dtype=np.int64).copy()
    if rate <= 0:
        return out
    mask = rng.random(out.size) < float(rate)
    if np.any(mask):
        jumps = rng.integers(1, q, size=int(mask.sum()), dtype=np.int64)
        out[mask] = (out[mask] + jumps) % int(q)
    return out


def _block_transform(q: int, n_blocks: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    if n_blocks is None:
        n_blocks = max(2, min(int(q), int(round(math.sqrt(q)))))
    n_blocks = max(1, min(int(n_blocks), int(q)))
    return lambda x: (np.asarray(x, dtype=np.int64) * n_blocks // int(q)).astype(np.int64)


def _canalizing_transform(q: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: (np.asarray(x, dtype=np.int64) != 0).astype(np.int64)


def _perm_transform(perm: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.asarray(perm, dtype=np.int64)[np.asarray(x, dtype=np.int64)]


def _force_B_label_availability(tc: TemporalChain) -> None:
    # Give B a non-collapsed cycling source.  Copying a predecessor is enough to
    # make the post-B boundary range over all q symbols when B initial states are all varied.
    for v in range(tc.offB, tc.offB + tc.kB):
        preds = tc.joint.preds[v]
        if preds:
            tc.joint.rules[v] = _rule_from_source(tc.q, len(preds), 0, None)


def _interface_targets(tc: TemporalChain) -> tuple[int, ...]:
    targets: list[int] = []
    targets.extend(tc.offB + j for j in range(tc.w))
    for pairs in (
        tc.interface_BC1,
        tc.interface_BD1,
        tc.interface_C1A1,
        tc.interface_D1A1,
        tc.interface_A1C2,
        tc.interface_A1D2,
        tc.interface_C2A2,
        tc.interface_D2A2,
    ):
        targets.extend(t for _, t in pairs)
    return tuple(sorted(set(int(x) for x in targets)))


def apply_rule_ensemble(
    tc: TemporalChain,
    ensemble: RuleEnsemble,
    rng: np.random.Generator,
    mutation_rate: float = 0.0,
    n_blocks: int | None = None,
    mutate_all_rules: bool = True,
) -> dict:
    q = tc.q
    transforms: dict = {}
    if ensemble == "random":
        transforms = dict(mode="random")
    else:
        _force_B_label_availability(tc)
        if ensemble == "copy":
            t1c = t1d = t2c = t2d = None
            transforms = dict(mode="copy")
        elif ensemble == "permutive":
            p1c = rng.permutation(q).astype(np.int64)
            p1d = rng.permutation(q).astype(np.int64)
            p2c = rng.permutation(q).astype(np.int64)
            p2d = rng.permutation(q).astype(np.int64)
            t1c, t1d, t2c, t2d = map(_perm_transform, (p1c, p1d, p2c, p2d))
            transforms = dict(mode="permutive", p1c=p1c.tolist(), p1d=p1d.tolist(), p2c=p2c.tolist(), p2d=p2d.tolist())
        elif ensemble == "block":
            # First convergence extracts a coarse block label.  The second
            # diamond should test transport of that already-extracted label,
            # so it copies the A1 panel rather than coarse-graining again.
            t1c = t1d = _block_transform(q, n_blocks)
            t2c = t2d = None
            transforms = dict(mode="block")
        elif ensemble == "canalizing":
            # Same logic as block: canalize once, then test transport.
            t1c = t1d = _canalizing_transform(q)
            t2c = t2d = None
            transforms = dict(mode="canalizing")
        elif ensemble == "constant":
            t1c = t1d = t2c = t2d = _block_transform(q, 1)
            transforms = dict(mode="constant")
        else:
            raise ValueError(f"unknown ensemble {ensemble!r}")

        # First diamond B -> C1/D1 -> A1.
        for (b, c1), (_, d1) in zip(tc.interface_BC1, tc.interface_BD1):
            if ensemble == "constant":
                _set_constant(tc, c1, 0); _set_constant(tc, d1, 0)
            else:
                _set_copy(tc, c1, b, t1c); _set_copy(tc, d1, b, t1d)
        for (c1, a1l), (d1, a1r) in zip(tc.interface_C1A1, tc.interface_D1A1):
            if ensemble == "constant":
                _set_constant(tc, a1l, 0); _set_constant(tc, a1r, 0)
            else:
                _set_copy(tc, a1l, c1, None); _set_copy(tc, a1r, d1, None)

        # Second diamond A1 panels -> C2/D2 -> A2.
        for (a1l, c2), (a1r, d2) in zip(tc.interface_A1C2, tc.interface_A1D2):
            if ensemble == "constant":
                _set_constant(tc, c2, 0); _set_constant(tc, d2, 0)
            else:
                _set_copy(tc, c2, a1l, t2c); _set_copy(tc, d2, a1r, t2d)
        for (c2, a2l), (d2, a2r) in zip(tc.interface_C2A2, tc.interface_D2A2):
            if ensemble == "constant":
                _set_constant(tc, a2l, 0); _set_constant(tc, a2r, 0)
            else:
                _set_copy(tc, a2l, c2, None); _set_copy(tc, a2r, d2, None)

    if ensemble != "random" and mutation_rate > 0:
        targets = range(tc.k_total) if mutate_all_rules else _interface_targets(tc)
        for v in targets:
            tc.joint.rules[int(v)] = _mutate_rule(tc.joint.rules[int(v)], q, mutation_rate, rng)

    tc.joint.meta["rule_ensemble"] = str(ensemble)
    tc.joint.meta["mutation_rate"] = float(mutation_rate)
    return transforms


def make_rule_temporal_chain(
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
) -> TemporalChain:
    tc = make_temporal_chain(kB, kM, kA, q, w, rng, extra_B=extra_B, extra_M=extra_M, extra_A=extra_A)
    transforms = apply_rule_ensemble(tc, rule_ensemble, rng, mutation_rate, n_blocks, mutate_all_rules)
    tc.meta.update(rule_ensemble=str(rule_ensemble), mutation_rate=float(mutation_rate), transforms=transforms)
    return tc


# --------------------------------------------------------------------------- #
# Schedules / initial states / measurement
# --------------------------------------------------------------------------- #
def canonical_schedules(tc: TemporalChain) -> dict[str, tuple[int, ...]]:
    B = tuple(range(tc.offB, tc.offC1))
    C1 = tuple(range(tc.offC1, tc.offD1))
    D1 = tuple(range(tc.offD1, tc.offA1))
    A1 = tuple(range(tc.offA1, tc.offC2))
    C2 = tuple(range(tc.offC2, tc.offD2))
    D2 = tuple(range(tc.offD2, tc.offA2))
    A2 = tuple(range(tc.offA2, tc.k_total))
    postB = B
    mid = B + C1 + D1 + A1
    final = mid + C2 + D2 + A2
    return dict(postB=postB, mid=mid, final=final)


def initial_states(tc: TemporalChain, mode: InitialMode = "source_all", n_random: int = 4096, rng: np.random.Generator | None = None) -> np.ndarray:
    q = tc.q
    if mode == "source_all":
        b_states = np.arange(q ** tc.kB, dtype=np.int64)
        return b_states.copy()  # downstream coordinates are zero, so encoded state is just B block.
    if mode == "source_random":
        rng = np.random.default_rng(0) if rng is None else rng
        b_states = rng.integers(0, q ** tc.kB, size=int(n_random), dtype=np.int64)
        return np.unique(b_states).astype(np.int64)
    if mode == "joint_random":
        rng = np.random.default_rng(0) if rng is None else rng
        n_total = int(q ** tc.k_total)
        states = rng.integers(0, n_total, size=int(n_random), dtype=np.int64)
        # Keep multiplicities out of the exact equalizer graph; this makes the
        # run deterministic for a given sample size and seed while still probing
        # downstream initial-memory noise.
        return np.unique(states).astype(np.int64)
    raise ValueError("initial mode must be source_all, source_random, or joint_random")


def temporal_chain_measure(
    tc: TemporalChain,
    initial_mode: InitialMode = "source_all",
    n_random_initial: int = 4096,
    schedule_mode: Literal["canonical"] = "canonical",
    source_for_liveness: Literal["initial", "postB"] = "postB",
    rng: np.random.Generator | None = None,
) -> dict:
    if schedule_mode != "canonical":
        raise ValueError("only canonical schedule is implemented in this module")
    states0 = initial_states(tc, mode=initial_mode, n_random=n_random_initial, rng=rng)
    sched = canonical_schedules(tc)
    states_postB = _step_subset(tc.joint, states0, sched["postB"])
    states_mid = _step_subset(tc.joint, states0, sched["mid"])
    states_final = _step_subset(tc.joint, states0, sched["final"])

    source_initial = _word(states0, tc.q, tc.b_boundary)
    source_postB = _word(states_postB, tc.q, tc.b_boundary)
    source = source_initial if source_for_liveness == "initial" else source_postB

    a1_left = _word(states_mid, tc.q, tc.a1_left)
    a1_right = _word(states_mid, tc.q, tc.a1_right)
    a2_left = _word(states_final, tc.q, tc.a2_left)
    a2_right = _word(states_final, tc.q, tc.a2_right)

    eq1 = exact_equalizer_labels(a1_left, a1_right, tc.q, tc.w)
    eq2 = exact_equalizer_labels(a2_left, a2_right, tc.q, tc.w)
    s1 = eq1["labels"]
    s2 = eq2["labels"]

    h_s1 = _entropy_discrete(s1)
    h_s2 = _entropy_discrete(s2)
    h_src = _entropy_discrete(source)
    mi_s1_s2 = _mutual_info_discrete(s1, s2)
    mi_src_s1 = _mutual_info_discrete(source, s1)
    mi_src_s2 = _mutual_info_discrete(source, s2)
    acc_s1_s2 = _best_map_accuracy(s1, s2)
    acc_s2_s1 = _best_map_accuracy(s2, s1)
    vi_norm = _variation_of_information_norm(s1, s2)

    # Candidate transport if both exact alphabets are nontrivial and S1 nearly determines S2.
    transport = bool(
        eq1["residual_qcoords"] > 1e-9
        and eq2["residual_qcoords"] > 1e-9
        and (mi_s1_s2 / max(h_s2, 1e-12)) > 0.8
        and acc_s1_s2 > 0.95
    )
    live_transport = bool(transport and (mi_src_s2 / max(h_s2, 1e-12)) > 0.25)

    return {
        "kB": int(tc.kB),
        "kM": int(tc.kM),
        "kA": int(tc.kA),
        "q": int(tc.q),
        "w": int(tc.w),
        "k_total": int(tc.k_total),
        "rule_ensemble": str(tc.meta.get("rule_ensemble", tc.joint.meta.get("rule_ensemble", "unknown"))),
        "mutation_rate": float(tc.meta.get("mutation_rate", tc.joint.meta.get("mutation_rate", 0.0))),
        "initial_mode": str(initial_mode),
        "n_initial_states": int(len(states0)),
        "source_for_liveness": str(source_for_liveness),
        "s1_shared_classes": int(eq1["shared_classes"]),
        "s2_shared_classes": int(eq2["shared_classes"]),
        "s1_residual_qcoords": float(eq1["residual_qcoords"]),
        "s2_residual_qcoords": float(eq2["residual_qcoords"]),
        "s1_edge_density": float(eq1["edge_density"]),
        "s2_edge_density": float(eq2["edge_density"]),
        "H_source_bits": float(h_src),
        "H_s1_bits": float(h_s1),
        "H_s2_bits": float(h_s2),
        "transport_mi_bits": float(mi_s1_s2),
        "transport_mi_over_Hs1": float(mi_s1_s2 / h_s1) if h_s1 > 1e-12 else 0.0,
        "transport_mi_over_Hs2": float(mi_s1_s2 / h_s2) if h_s2 > 1e-12 else 0.0,
        "source_to_s1_mi_bits": float(mi_src_s1),
        "source_to_s2_mi_bits": float(mi_src_s2),
        "source_to_s1_mi_over_Hs1": float(mi_src_s1 / h_s1) if h_s1 > 1e-12 else 0.0,
        "source_to_s2_mi_over_Hs2": float(mi_src_s2 / h_s2) if h_s2 > 1e-12 else 0.0,
        "s1_to_s2_best_accuracy": float(acc_s1_s2),
        "s2_to_s1_best_accuracy": float(acc_s2_s1),
        "s1_s2_variation_of_information_norm": float(vi_norm),
        "transport_selected": bool(transport),
        "live_transport_selected": bool(live_transport),
    }


# --------------------------------------------------------------------------- #
# Sweeps / analysis / plotting
# --------------------------------------------------------------------------- #
def _stable_ensemble_code(name: str) -> int:
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(str(name))) % 10007)


def run_temporal_chain_sweep(
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2,),
    ws: Iterable[int] = (1,),
    q: int = 4,
    n_instances: int = 20,
    rule_ensembles: Iterable[RuleEnsemble] = ("random", "copy", "permutive", "block", "canalizing", "constant"),
    mutation_rates: Iterable[float] = (0.0,),
    initial_mode: InitialMode = "source_all",
    source_for_liveness: Literal["initial", "postB"] = "postB",
    n_random_initial: int = 4096,
    n_blocks: int | None = None,
    mutate_all_rules: bool = True,
    extra_B: int | None = None,
    extra_M: int | None = None,
    extra_A: int | None = None,
    base_seed: int = 0,
    max_encoded_states: int = 10**12,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for kB in ks_B:
        for kM in ks_M:
            for kA in ks_A:
                k_total = kB + 2 * kM + kA + 2 * kM + kA
                if q ** k_total > max_encoded_states:
                    if verbose:
                        print(f"note: encoded state space q^k={q}^{k_total} is large; subset simulation is still used", flush=True)
                for w in ws:
                    if w > min(kB, kM) or kA < 2 * w:
                        continue
                    for ens in rule_ensembles:
                        for mu in mutation_rates:
                            for inst in range(n_instances):
                                seed = (
                                    base_seed * 1000003
                                    + kB * 524287
                                    + kM * 8191
                                    + kA * 257
                                    + w * 31
                                    + _stable_ensemble_code(str(ens)) * 1009
                                    + int(round(float(mu) * 10000)) * 17
                                    + inst
                                ) % 2**32
                                rng = np.random.default_rng(seed)
                                tc = make_rule_temporal_chain(
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
                                m = temporal_chain_measure(
                                    tc,
                                    initial_mode=initial_mode,
                                    n_random_initial=int(n_random_initial),
                                    source_for_liveness=source_for_liveness,
                                    rng=rng,
                                )
                                m.update(seed=int(seed), instance=int(inst))
                                rows.append(m)
                            if verbose:
                                print(f"temporal-chain ensemble={ens}, mu={mu:g}, kB={kB}, kM={kM}, kA={kA}, w={w} done", flush=True)
    return pd.DataFrame(rows)


def analyze_temporal_chain(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    by = []
    for keys, sub in df.groupby(["rule_ensemble", "mutation_rate"]):
        ens, mu = keys
        by.append(
            dict(
                rule_ensemble=str(ens),
                mutation_rate=float(mu),
                n=int(len(sub)),
                s1_residual=float(sub["s1_residual_qcoords"].mean()),
                s2_residual=float(sub["s2_residual_qcoords"].mean()),
                transport_fraction=float(sub["transport_selected"].mean()),
                live_transport_fraction=float(sub["live_transport_selected"].mean()),
                transport_mi_over_Hs2=float(sub["transport_mi_over_Hs2"].mean()),
                source_to_s2_mi_over_Hs2=float(sub["source_to_s2_mi_over_Hs2"].mean()),
                s1_to_s2_best_accuracy=float(sub["s1_to_s2_best_accuracy"].mean()),
                vi_norm=float(sub["s1_s2_variation_of_information_norm"].mean()),
            )
        )
    zero = df[np.isclose(df["mutation_rate"].astype(float), 0.0)]
    random_zero = zero[zero["rule_ensemble"] == "random"]
    structured_zero = zero[(zero["rule_ensemble"] != "random") & (zero["rule_ensemble"] != "constant")]
    random_live = float(random_zero["live_transport_selected"].mean()) if len(random_zero) else math.nan
    structured_live = float(structured_zero["live_transport_selected"].mean()) if len(structured_zero) else math.nan
    structured_transport = float(structured_zero["transport_selected"].mean()) if len(structured_zero) else math.nan
    mean_structured_source_to_s2 = float(structured_zero["source_to_s2_mi_over_Hs2"].mean()) if len(structured_zero) else math.nan
    mean_random_source_to_s2 = float(random_zero["source_to_s2_mi_over_Hs2"].mean()) if len(random_zero) else math.nan

    if not math.isnan(structured_live) and structured_live > 0.6 and (math.isnan(random_live) or random_live < 0.1):
        verdict = "TEMPORAL LIVE-TRANSPORT SIGNAL: structured chains carry a live label through two convergences while random chains do not"
    elif not math.isnan(structured_transport) and structured_transport > 0.6:
        verdict = "TEMPORAL TRANSPORT WITHOUT STRONG SOURCE-LIVENESS: S1 maps to S2, but source control needs inspection"
    elif not math.isnan(structured_live) and structured_live > (0 if math.isnan(random_live) else random_live) + 0.1:
        verdict = "PARTIAL TEMPORAL-CHAIN SIGNAL: selected transport exists but is not robust/generic"
    else:
        verdict = "NO GENERIC TEMPORAL TRANSPORT in this regime"
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        random_live_transport_fraction_at_zero_mutation=random_live,
        structured_live_transport_fraction_at_zero_mutation=structured_live,
        structured_transport_fraction_at_zero_mutation=structured_transport,
        mean_structured_source_to_s2_mi_over_Hs2=mean_structured_source_to_s2,
        mean_random_source_to_s2_mi_over_Hs2=mean_random_source_to_s2,
        by_ensemble=by,
    )


def plot_temporal_chain(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for ens in sorted(df.rule_ensemble.unique()):
        sub = df[df.rule_ensemble == ens]
        g = sub.groupby("mutation_rate")["transport_mi_over_Hs2"].mean().dropna()
        ax[0].plot(g.index, g.values, "o-", label=str(ens))
    ax[0].set_xlabel("mutation rate")
    ax[0].set_ylabel(r"$I(S_1;S_2)/H(S_2)$")
    ax[0].set_title("label transport through chain")
    ax[0].legend(fontsize=8)

    for ens in sorted(df.rule_ensemble.unique()):
        sub = df[df.rule_ensemble == ens]
        g = sub.groupby("mutation_rate")["source_to_s2_mi_over_Hs2"].mean().dropna()
        ax[1].plot(g.index, g.values, "o-", label=str(ens))
    ax[1].set_xlabel("mutation rate")
    ax[1].set_ylabel(r"$I(source;S_2)/H(S_2)$")
    ax[1].set_title("source-liveness at final label")
    ax[1].legend(fontsize=8)

    for ens in sorted(df.rule_ensemble.unique()):
        sub = df[df.rule_ensemble == ens]
        g = sub.groupby("mutation_rate")["live_transport_selected"].mean().dropna()
        ax[2].plot(g.index, g.values, "o-", label=str(ens))
    ax[2].set_xlabel("mutation rate")
    ax[2].set_ylabel("fraction")
    ax[2].set_title("live-transport selected")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_ensembles(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("q", type=int, nargs="?", default=4)
    ap.add_argument("--ks-B", default="2")
    ap.add_argument("--ks-M", default="2")
    ap.add_argument("--ks-A", default="2")
    ap.add_argument("--ws", default="1")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--rule-ensembles", default="random,copy,permutive,block,canalizing,constant")
    ap.add_argument("--mutation-rates", default="0")
    ap.add_argument("--initial-mode", choices=("source_all", "source_random", "joint_random"), default="source_all")
    ap.add_argument("--n-random-initial", type=int, default=4096, help="number of initial states for source_random/joint_random modes")
    ap.add_argument("--source-for-liveness", choices=("initial", "postB"), default="postB")
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--interface-only-mutation", action="store_true", help="mutate only declared interface rules, not all rules")
    ap.add_argument("--extra-B", type=int, default=None)
    ap.add_argument("--extra-M", type=int, default=None)
    ap.add_argument("--extra-A", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", default="example_results/temporal_chain.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df = run_temporal_chain_sweep(
        ks_B=_parse_ints(args.ks_B),
        ks_M=_parse_ints(args.ks_M),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        q=int(args.q),
        n_instances=int(args.instances),
        rule_ensembles=_parse_ensembles(args.rule_ensembles),  # type: ignore[arg-type]
        mutation_rates=_parse_floats(args.mutation_rates),
        initial_mode=args.initial_mode,  # type: ignore[arg-type]
        source_for_liveness=args.source_for_liveness,  # type: ignore[arg-type]
        n_random_initial=int(args.n_random_initial),
        n_blocks=args.n_blocks,
        mutate_all_rules=not bool(args.interface_only_mutation),
        extra_B=args.extra_B,
        extra_M=args.extra_M,
        extra_A=args.extra_A,
        base_seed=int(args.base_seed),
        verbose=not bool(args.quiet),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = analyze_temporal_chain(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    if args.plot:
        plot_temporal_chain(df, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2, default=float))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
