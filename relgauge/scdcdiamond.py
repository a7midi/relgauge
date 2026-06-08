r"""
scdcdiamond.py -- SCDC diamond quotient-cost diagnostics.

This module follows diamond.py / diamondphase.py, but it tests a different
object.  The earlier diamond experiment asked whether a post-selected herald in

        B
       / \
      C   D
       \ /
        A

naturally correlates the branch phases.  In random finite SCC diamonds that was
negative: the herald was usually insensitive to the phases.

Here we ask the SCDC question instead:

    How much local quotienting is required to make a declared diamond
    consistency condition true as a law rather than as a post-selection event?

The implemented quotient is a finite local alphabet congruence profile.  It is
seeded by one of three constraints and then closed under local-rule
admissibility, mirroring the least-closure style of SCDC:

  branch_order
      Pair each C-before-D schedule with the matching D-before-C schedule and
      merge output alphabet values that differ.  In the feed-forward diamond
      this should usually cost zero: the raw branch updates commute exactly.
      This is a useful sanity check.

  sink_equal
      Force the two branch panels at A to agree modulo a shared local face
      quotient.  This converts the old herald s=(A_C_panel == A_D_panel) into a
      deterministic quotient law and measures the collapse required to do so.

  strict_schedule
      Force all admissible schedules to produce the same quotient output.  This
      is the aggressive strict schedule-gauge quotient; it is an upper-bound /
      control, not the preferred physical object.

Main reported quantity:

    quotient_cost_qcoords = sum_v log_q(q / number_of_classes_at_v)

i.e. the number of q-ary local coordinates removed by the least quotient.

Interpretation:
  * branch_order cost ~ 0 means the raw diamond branch order is already
    confluent, so phase laws do not come from bare branch-order noncommutation.
  * sink_equal cost large means a random diamond can be made consistent only by
    trivializing some sink-facing degrees of freedom.
  * sink_equal cost small/nonzero and structured would be the signal that SCDC
    extracts a nontrivial law rather than mere collapse.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import core as C

ConstraintMode = Literal["branch_order", "sink_equal", "strict_schedule"]
InitialPool = Literal["all", "rec"]
ObserverInit = Literal["zero", "all"]


# --------------------------------------------------------------------------- #
# Small union-find / quotient profile
# --------------------------------------------------------------------------- #
class UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        r = int(x)
        while self.p[r] != r:
            r = self.p[r]
        x = int(x)
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True

    def nclasses(self) -> int:
        return len({self.find(i) for i in range(len(self.p))})


class LocalProfile:
    """Local alphabet quotient profile, with optional tied vertex pairs.

    If vertices u and v are tied, they share the same UF object; a merge at one
    vertex is also a merge at the other.  This is how sink_equal represents two
    panels as the same logical face alphabet.
    """

    def __init__(self, k: int, q: int, tied_pairs: Iterable[tuple[int, int]] = ()):  # noqa: D401
        self.k = int(k)
        self.q = int(q)
        self.uf: list[UF] = [UF(q) for _ in range(k)]
        for a, b in tied_pairs:
            shared = UF(q)
            self.uf[int(a)] = shared
            self.uf[int(b)] = shared

    def union(self, v: int, a: int, b: int) -> bool:
        return self.uf[int(v)].union(int(a), int(b))

    def label(self, v: int, a: int) -> int:
        return self.uf[int(v)].find(int(a))

    def nclasses_by_vertex(self) -> list[int]:
        return [u.nclasses() for u in self.uf]

    def labels_by_vertex(self) -> list[list[int]]:
        return [[u.find(i) for i in range(self.q)] for u in self.uf]


# --------------------------------------------------------------------------- #
# Diamond construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SCDCDiamond:
    """Four-SCC feed-forward diamond B -> {C,D} -> A.

    Vertex layout:
      B: offB .. offC-1
      C: offC .. offD-1
      D: offD .. offA-1
      A: offA .. k_total-1

    A has two panels of width w:
      A[0:w] receives C
      A[w:2w] receives D
    so kA must be at least 2*w.
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
    def panel_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((self.offA + j, self.offA + self.w + j) for j in range(self.w))


def _random_rule(q: int, indeg: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, q, size=q ** indeg, dtype=np.int64)


def make_scdc_diamond(
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
) -> SCDCDiamond:
    if min(kB, kC, kD, kA) < 1:
        raise ValueError("all component sizes must be positive")
    if q < 2:
        raise ValueError("q must be at least 2")
    if w <= 0:
        raise ValueError("w must be positive")
    if w > min(kB, kC, kD):
        raise ValueError("w must be <= min(kB,kC,kD)")
    if kA < 2 * w:
        raise ValueError("sink panels require kA >= 2*w")
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
        b_src = offB + j
        c_tgt = offC + j
        d_tgt = offD + j
        pred_sets[c_tgt].add(b_src)
        pred_sets[d_tgt].add(b_src)
        interface_BC.append((b_src, c_tgt))
        interface_BD.append((b_src, d_tgt))

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
        if v < offC:
            rules.append(B.rules[v].copy())
        else:
            rules.append(_random_rule(q, len(preds[v]), rng))

    joint = C.RelationalSystem(
        k_total,
        q,
        preds,
        rules,
        meta={
            "ensemble": "scdc_diamond",
            "kB": int(kB),
            "kC": int(kC),
            "kD": int(kD),
            "kA": int(kA),
            "w": int(w),
            "extra_B": int(extra_B),
            "extra_C": int(extra_C),
            "extra_D": int(extra_D),
            "extra_A": int(extra_A),
        },
    )
    return SCDCDiamond(
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


@dataclass(frozen=True)
class ScheduleInfo:
    schedule: tuple[int, ...]
    pair_id: int
    branch_order: str


def admissible_schedules(ds: SCDCDiamond) -> list[ScheduleInfo]:
    Bv = tuple(range(ds.offB, ds.offC))
    Cv = tuple(range(ds.offC, ds.offD))
    Dv = tuple(range(ds.offD, ds.offA))
    Av = tuple(range(ds.offA, ds.offA + ds.kA))
    out: list[ScheduleInfo] = []
    pair_id = 0
    for pB in itertools.permutations(Bv):
        for pC in itertools.permutations(Cv):
            for pD in itertools.permutations(Dv):
                for pA in itertools.permutations(Av):
                    out.append(ScheduleInfo(tuple(pB + pC + pD + pA), pair_id, "CD"))
                    out.append(ScheduleInfo(tuple(pB + pD + pC + pA), pair_id, "DC"))
                    pair_id += 1
    return out


def step_maps(ds: SCDCDiamond) -> tuple[np.ndarray, list[ScheduleInfo]]:
    infos = admissible_schedules(ds)
    maps = np.empty((len(infos), ds.n_joint_states), dtype=np.int64)
    for i, info in enumerate(infos):
        maps[i] = ds.joint.step_map(info.schedule)
    return maps, infos


def _component_pool(sys: C.RelationalSystem, mode: InitialPool) -> np.ndarray:
    if mode == "all":
        return np.arange(sys.q ** sys.k, dtype=np.int64)
    if mode == "rec":
        sm = sys.all_step_maps()
        adj = C.orbit_adjacency_fast(sm)
        rec = C.recurrent_states(adj)
        if len(rec) == 0:
            return np.arange(sys.q ** sys.k, dtype=np.int64)
        return rec.astype(np.int64)
    raise ValueError("initial pool must be 'all' or 'rec'")


def _initial_states(ds: SCDCDiamond, pool: InitialPool = "rec", observer_init: ObserverInit = "zero") -> np.ndarray:
    b = _component_pool(ds.B, pool)
    c = _component_pool(ds.Csys, pool)
    d = _component_pool(ds.Dsys, pool)
    if observer_init == "zero":
        a = np.array([0], dtype=np.int64)
    elif observer_init == "all":
        a = np.arange(ds.q ** ds.kA, dtype=np.int64)
    else:
        raise ValueError("observer_init must be 'zero' or 'all'")
    if len(b) * len(c) * len(d) * len(a) == 0:
        return np.array([], dtype=np.int64)
    bb = np.repeat(b, len(c) * len(d) * len(a))
    cc = np.tile(np.repeat(c, len(d) * len(a)), len(b))
    dd = np.tile(np.repeat(d, len(a)), len(b) * len(c))
    aa = np.tile(a, len(b) * len(c) * len(d))
    return np.unique(
        bb
        + cc * (ds.q ** ds.kB)
        + dd * (ds.q ** (ds.kB + ds.kC))
        + aa * (ds.q ** (ds.kB + ds.kC + ds.kD))
    ).astype(np.int64)


def _digit(indices: np.ndarray, q: int, vertex: int) -> np.ndarray:
    return (np.asarray(indices, dtype=np.int64) // (q ** int(vertex))) % q


def _merge_digit_pairs(profile: LocalProfile, v: int, a_vals: np.ndarray, b_vals: np.ndarray) -> int:
    if len(a_vals) == 0:
        return 0
    pairs = np.unique(np.stack([a_vals.astype(np.int64), b_vals.astype(np.int64)], axis=1), axis=0)
    changed = 0
    for a, b in pairs:
        if profile.union(v, int(a), int(b)):
            changed += 1
    return changed


def _seed_branch_order(profile: LocalProfile, ds: SCDCDiamond, maps: np.ndarray, infos: list[ScheduleInfo], states: np.ndarray) -> int:
    by_pair: dict[int, dict[str, int]] = {}
    for i, info in enumerate(infos):
        by_pair.setdefault(info.pair_id, {})[info.branch_order] = i
    changed = 0
    for pair in by_pair.values():
        if "CD" not in pair or "DC" not in pair:
            continue
        y_cd = maps[pair["CD"], states]
        y_dc = maps[pair["DC"], states]
        for v in range(ds.k_total):
            changed += _merge_digit_pairs(profile, v, _digit(y_cd, ds.q, v), _digit(y_dc, ds.q, v))
    return changed


def _seed_strict_schedule(profile: LocalProfile, ds: SCDCDiamond, maps: np.ndarray, states: np.ndarray) -> int:
    if maps.shape[0] < 2 or len(states) == 0:
        return 0
    base = maps[0, states]
    changed = 0
    for i in range(1, maps.shape[0]):
        cur = maps[i, states]
        for v in range(ds.k_total):
            changed += _merge_digit_pairs(profile, v, _digit(base, ds.q, v), _digit(cur, ds.q, v))
    return changed


def _seed_sink_equal(profile: LocalProfile, ds: SCDCDiamond, maps: np.ndarray, states: np.ndarray) -> int:
    changed = 0
    if len(states) == 0:
        return changed
    finals = maps[:, states].reshape(-1)
    for left, right in ds.panel_pairs:
        changed += _merge_digit_pairs(profile, left, _digit(finals, ds.q, left), _digit(finals, ds.q, right))
    return changed


def _admissibility_closure(profile: LocalProfile, sys: C.RelationalSystem, max_iter: int = 1000) -> tuple[int, int]:
    """Close under local quotient semantics.

    If predecessor input tuples have the same quotient signature, their local
    outputs must be equivalent at that vertex.
    """
    total_merges = 0
    iterations = 0
    tuples_by_v = []
    for v in range(sys.k):
        d = len(sys.preds[v])
        tuples_by_v.append(C.all_states(d, sys.q) if d else np.zeros((1, 0), dtype=np.int64))

    changed = True
    while changed:
        changed = False
        iterations += 1
        if iterations > max_iter:
            raise RuntimeError("admissibility closure did not converge")
        for v in range(sys.k):
            preds = sys.preds[v]
            if not preds:
                continue
            groups: dict[tuple[int, ...], list[int]] = {}
            tuples = tuples_by_v[v]
            for row in range(tuples.shape[0]):
                vals = tuples[row]
                sig = tuple(profile.label(preds[j], int(vals[j])) for j in range(len(preds)))
                out = int(sys.rules[v][row])
                groups.setdefault(sig, []).append(out)
            for outs in groups.values():
                head = outs[0]
                for out in outs[1:]:
                    if profile.union(v, head, out):
                        total_merges += 1
                        changed = True
    return total_merges, iterations


def quotient_cost(profile: LocalProfile) -> dict:
    q = profile.q
    ncls = profile.nclasses_by_vertex()
    costs_q = [math.log(q / n, q) for n in ncls]
    costs_bits = [math.log2(q / n) for n in ncls]
    return {
        "classes_by_vertex": json.dumps([int(n) for n in ncls]),
        "vertex_costs_qcoords": json.dumps([float(x) for x in costs_q]),
        "quotient_dim_qcoords": float(sum(math.log(n, q) for n in ncls)),
        "quotient_cost_qcoords": float(sum(costs_q)),
        "quotient_cost_bits": float(sum(costs_bits)),
        "collapsed_vertices": int(sum(1 for n in ncls if n < q)),
        "fully_collapsed_vertices": int(sum(1 for n in ncls if n == 1)),
    }


def _range_costs(profile: LocalProfile, vertices: Iterable[int]) -> tuple[float, float]:
    q = profile.q
    c_q = 0.0
    c_b = 0.0
    ncls = profile.nclasses_by_vertex()
    for v in vertices:
        n = ncls[int(v)]
        c_q += math.log(q / n, q)
        c_b += math.log2(q / n)
    return c_q, c_b


def scdc_diamond_quotient_cost(
    ds: SCDCDiamond,
    constraint_mode: ConstraintMode = "sink_equal",
    initial_pool: InitialPool = "rec",
    observer_init: ObserverInit = "zero",
    close_admissibility: bool = True,
    precomputed_maps: tuple[np.ndarray, list[ScheduleInfo]] | None = None,
) -> dict:
    """Compute least local quotient cost for a diamond constraint."""
    if constraint_mode not in ("branch_order", "sink_equal", "strict_schedule"):
        raise ValueError(f"unknown constraint_mode: {constraint_mode!r}")
    if precomputed_maps is None:
        maps, infos = step_maps(ds)
    else:
        maps, infos = precomputed_maps
    states = _initial_states(ds, initial_pool, observer_init)

    tied = ds.panel_pairs if constraint_mode == "sink_equal" else ()
    profile = LocalProfile(ds.k_total, ds.q, tied_pairs=tied)

    branch_merges = 0
    sink_merges = 0
    schedule_merges = 0
    if constraint_mode == "branch_order":
        branch_merges = _seed_branch_order(profile, ds, maps, infos, states)
    elif constraint_mode == "sink_equal":
        sink_merges = _seed_sink_equal(profile, ds, maps, states)
    elif constraint_mode == "strict_schedule":
        schedule_merges = _seed_strict_schedule(profile, ds, maps, states)

    adm_merges = 0
    adm_iters = 0
    if close_admissibility:
        adm_merges, adm_iters = _admissibility_closure(profile, ds.joint)

    out = quotient_cost(profile)
    ranges = {
        "B": range(ds.offB, ds.offC),
        "C": range(ds.offC, ds.offD),
        "D": range(ds.offD, ds.offA),
        "A": range(ds.offA, ds.offA + ds.kA),
        "A_panel": [v for pair in ds.panel_pairs for v in pair],
    }
    for name, vertices in ranges.items():
        cq, cb = _range_costs(profile, vertices)
        out[f"cost_{name}_qcoords"] = float(cq)
        out[f"cost_{name}_bits"] = float(cb)

    out.update(
        constraint_mode=constraint_mode,
        initial_pool=initial_pool,
        observer_init=observer_init,
        close_admissibility=bool(close_admissibility),
        kB=int(ds.kB),
        kC=int(ds.kC),
        kD=int(ds.kD),
        kA=int(ds.kA),
        kM=int(ds.kC) if ds.kC == ds.kD else None,
        k_total=int(ds.k_total),
        q=int(ds.q),
        w=int(ds.w),
        n_initial_states=int(len(states)),
        n_schedules=int(maps.shape[0]),
        branch_order_seed_merges=int(branch_merges),
        sink_seed_merges=int(sink_merges),
        strict_schedule_seed_merges=int(schedule_merges),
        admissibility_merges=int(adm_merges),
        admissibility_iterations=int(adm_iters),
        extra_B=int(ds.meta["extra_B"]),
        extra_C=int(ds.meta["extra_C"]),
        extra_D=int(ds.meta["extra_D"]),
        extra_A=int(ds.meta["extra_A"]),
    )
    return out


# --------------------------------------------------------------------------- #
# Sweeps, analysis, plotting, CLI
# --------------------------------------------------------------------------- #
def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _parse_modes(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def run_scdc_diamond_sweep(
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2,),
    ws: Iterable[int] = (1,),
    q: int = 4,
    n_instances: int = 20,
    constraint_modes: Iterable[ConstraintMode] = ("branch_order", "sink_equal", "strict_schedule"),
    initial_pool: InitialPool = "rec",
    observer_init: ObserverInit = "zero",
    close_admissibility: bool = True,
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
                    for inst in range(n_instances):
                        seed = (base_seed * 1000003 + kB * 5003 + kM * 809 + kA * 131 + w * 19 + inst) % 2**32
                        rng = np.random.default_rng(seed)
                        ds = make_scdc_diamond(
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
                        maps_infos = step_maps(ds)
                        for mode in constraint_modes:
                            m = scdc_diamond_quotient_cost(
                                ds,
                                constraint_mode=mode,  # type: ignore[arg-type]
                                initial_pool=initial_pool,
                                observer_init=observer_init,
                                close_admissibility=close_admissibility,
                                precomputed_maps=maps_infos,
                            )
                            m.update(seed=int(seed), instance=int(inst))
                            rows.append(m)
                    if verbose:
                        print(f"SCDC-diamond kB={kB}, kM={kM}, kA={kA}, w={w} done", flush=True)
    return pd.DataFrame(rows)


def analyze_scdc_diamond(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    eps = 1e-9
    bound_viol = int((df["quotient_cost_qcoords"] < -eps).sum())
    max_cost = df["k_total"].astype(float)
    bound_viol += int((df["quotient_cost_qcoords"] > max_cost + eps).sum())

    means = {
        "mean_cost_qcoords": float(df["quotient_cost_qcoords"].mean()),
        "mean_cost_bits": float(df["quotient_cost_bits"].mean()),
        "mean_collapsed_vertices": float(df["collapsed_vertices"].mean()),
        "mean_admissibility_merges": float(df["admissibility_merges"].mean()),
    }
    mode_means = (
        df.groupby("constraint_mode")
        .agg(
            cost_qcoords=("quotient_cost_qcoords", "mean"),
            cost_bits=("quotient_cost_bits", "mean"),
            cost_A_qcoords=("cost_A_qcoords", "mean"),
            cost_B_qcoords=("cost_B_qcoords", "mean"),
            cost_C_qcoords=("cost_C_qcoords", "mean"),
            cost_D_qcoords=("cost_D_qcoords", "mean"),
            collapsed_vertices=("collapsed_vertices", "mean"),
            seed_merges_sink=("sink_seed_merges", "mean"),
            seed_merges_order=("branch_order_seed_merges", "mean"),
            seed_merges_schedule=("strict_schedule_seed_merges", "mean"),
        )
        .reset_index()
    )
    by_mode = {str(r.constraint_mode): {k: float(getattr(r, k)) for k in mode_means.columns if k != "constraint_mode"} for r in mode_means.itertuples(index=False)}

    scaling = []
    group_cols = ["constraint_mode", "kM", "kA", "w"]
    for keys, sub in df.groupby(group_cols):
        mode, kM, kA, w = keys
        by_k = sub.groupby("kB")["quotient_cost_qcoords"].mean()
        scaling.append(
            dict(
                constraint_mode=str(mode),
                kM=int(kM),
                kA=int(kA),
                w=int(w),
                cost_by_kB={int(k): float(v) for k, v in by_k.items()},
            )
        )

    order_cost = by_mode.get("branch_order", {}).get("cost_qcoords", math.nan)
    sink_cost = by_mode.get("sink_equal", {}).get("cost_qcoords", math.nan)
    strict_cost = by_mode.get("strict_schedule", {}).get("cost_qcoords", math.nan)
    if bound_viol:
        verdict = "IMPLEMENTATION WARNING: quotient cost bound violated"
    elif not math.isnan(order_cost) and order_cost < 1e-6 and not math.isnan(sink_cost) and sink_cost > 0.25:
        verdict = "SCDC COST SIGNAL: raw branch order is confluent, but sink consistency requires quotient collapse"
    elif not math.isnan(order_cost) and order_cost < 1e-6 and (math.isnan(sink_cost) or sink_cost <= 0.25):
        verdict = "LOW / TRIVIAL DIAMOND COST in this regime"
    elif not math.isnan(strict_cost) and strict_cost > 0.5:
        verdict = "STRICT SCHEDULE QUOTIENT IS COSTLY; inspect whether this is over-collapse"
    else:
        verdict = "MIXED / WEAK SCDC DIAMOND COST: inspect modes and components"

    return dict(
        verdict=verdict,
        cost_bound_violations=int(bound_viol),
        **means,
        by_mode=by_mode,
        scaling=scaling,
    )


def plot_scdc_diamond(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for mode in sorted(df.constraint_mode.unique()):
        sub = df[df.constraint_mode == mode]
        g = sub.groupby("w")["quotient_cost_qcoords"].mean().dropna()
        ax[0].plot(g.index, g.values, "o-", label=str(mode))
    ax[0].set_xlabel("interface width w")
    ax[0].set_ylabel("quotient cost (q-ary coords)")
    ax[0].set_title("SCDC diamond quotient cost")
    ax[0].legend(fontsize=8)

    comp_cols = ["cost_B_qcoords", "cost_C_qcoords", "cost_D_qcoords", "cost_A_qcoords"]
    g2 = df.groupby("constraint_mode")[comp_cols].mean()
    x = np.arange(len(g2.index))
    bottom = np.zeros(len(g2.index))
    for col in comp_cols:
        vals = g2[col].to_numpy(dtype=float)
        ax[1].bar(x, vals, bottom=bottom, label=col.replace("cost_", "").replace("_qcoords", ""))
        bottom += vals
    ax[1].set_xticks(x, list(g2.index), rotation=25, ha="right")
    ax[1].set_ylabel("component cost")
    ax[1].set_title("where the quotient collapses")
    ax[1].legend(fontsize=8)

    for mode in sorted(df.constraint_mode.unique()):
        sub = df[df.constraint_mode == mode]
        g = sub.groupby("kB")["quotient_cost_qcoords"].mean().dropna()
        ax[2].plot(g.index, g.values, "o-", label=str(mode))
    ax[2].set_xlabel("source bulk kB")
    ax[2].set_ylabel("quotient cost")
    ax[2].set_title("cost vs source bulk")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("q", type=int, nargs="?", default=4)
    ap.add_argument("--ks-B", default="2")
    ap.add_argument("--ks-M", default="2")
    ap.add_argument("--ks-A", default="2")
    ap.add_argument("--ws", default="1")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--constraint-modes", default="branch_order,sink_equal,strict_schedule")
    ap.add_argument("--initial-pool", choices=("all", "rec"), default="rec")
    ap.add_argument("--observer-init", choices=("zero", "all"), default="zero")
    ap.add_argument("--no-admissibility", action="store_true")
    ap.add_argument("--extra-B", type=int, default=None)
    ap.add_argument("--extra-M", type=int, default=None)
    ap.add_argument("--extra-A", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--max-joint-states", type=int, default=4 ** 9)
    ap.add_argument("--out", default="example_results/scdc_diamond.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df = run_scdc_diamond_sweep(
        ks_B=_parse_ints(args.ks_B),
        ks_M=_parse_ints(args.ks_M),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        q=args.q,
        n_instances=args.instances,
        constraint_modes=_parse_modes(args.constraint_modes),  # type: ignore[arg-type]
        initial_pool=args.initial_pool,  # type: ignore[arg-type]
        observer_init=args.observer_init,  # type: ignore[arg-type]
        close_admissibility=not args.no_admissibility,
        extra_B=args.extra_B,
        extra_M=args.extra_M,
        extra_A=args.extra_A,
        base_seed=args.base_seed,
        max_joint_states=args.max_joint_states,
        verbose=not args.quiet,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    res = analyze_scdc_diamond(df)
    print(json.dumps(res, indent=2, default=float))
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    if args.plot:
        plot_scdc_diamond(df, args.plot)
        print(f"wrote {args.plot}")
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
