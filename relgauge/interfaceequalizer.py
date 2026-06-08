"""
interfaceequalizer.py -- relational/equalizer version of SCDC diamond consistency.

Why this exists
---------------
The product-local SCDC diamond test (relgauge.scdcdiamond) showed that forcing
A's two sink panels to be equal using only coordinatewise alphabet quotients
usually collapses the whole random sink observer A.  That is expected: the
constraint

    A_left == A_right

is not naturally a product quotient.  It is a relation/equalizer between two
interface coordinates.  A product quotient can only identify values inside each
coordinate; a true interface law should preserve a shared boundary variable S:

    f_left(A_left) = f_right(A_right) in S.

This module computes the maximal finite shared alphabet S induced by the
reachable sink-panel pairs of a B -> {C,D} -> A diamond.  It reports the
interface residual

    residual = log_q |S|

and the corresponding equalizer cost

    cost = 2*w - residual

in q-ary coordinates.  For w=1:

    * perfect diagonal equality: |S|=q, residual=1, cost=1
    * trivial collapse:          |S|=1, residual=0, cost=2

The experiment compares this relational/equalizer cost with the old product
local-quotient cost on the same random diamond instances.
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

from . import core as C
from . import scdcdiamond as S

EqualizerMode = Literal["pairwise", "joint"]
InitialPool = S.InitialPool
ObserverInit = S.ObserverInit


# --------------------------------------------------------------------------- #
# Union-find and equalizer kernels
# --------------------------------------------------------------------------- #
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


def _encode_word(vals: np.ndarray, q: int) -> np.ndarray:
    """Encode an (n,w) word array over [q] into [q**w]."""
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


@dataclass(frozen=True)
class EqualizerSummary:
    """Maximal equalizer f_L(left)=f_R(right) for observed left/right words."""

    q: int
    width: int
    alphabet_size_per_side: int
    n_observed_pairs: int
    edge_density: float
    shared_classes: int
    residual_qcoords: float
    residual_bits: float
    cost_qcoords: float
    cost_bits: float
    class_sizes_left: tuple[int, ...]
    class_sizes_right: tuple[int, ...]
    observed_left_words: int
    observed_right_words: int

    def as_dict(self, prefix: str = "") -> dict:
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}alphabet_size_per_side": int(self.alphabet_size_per_side),
            f"{p}n_observed_pairs": int(self.n_observed_pairs),
            f"{p}edge_density": float(self.edge_density),
            f"{p}shared_classes": int(self.shared_classes),
            f"{p}residual_qcoords": float(self.residual_qcoords),
            f"{p}residual_bits": float(self.residual_bits),
            f"{p}cost_qcoords": float(self.cost_qcoords),
            f"{p}cost_bits": float(self.cost_bits),
            f"{p}class_sizes_left": json.dumps([int(x) for x in self.class_sizes_left]),
            f"{p}class_sizes_right": json.dumps([int(x) for x in self.class_sizes_right]),
            f"{p}observed_left_words": int(self.observed_left_words),
            f"{p}observed_right_words": int(self.observed_right_words),
        }


def maximal_shared_equalizer(left_words: np.ndarray, right_words: np.ndarray, q: int, width: int) -> EqualizerSummary:
    """Compute the maximal shared alphabet S for observed word pairs.

    The observed pairs define a bipartite graph on

        L = [0, q**width), R = [0, q**width).

    For every observed pair (l,r), the equalizer condition requires
    f_L(l)=f_R(r).  The maximal such shared alphabet is the connected-component
    quotient of the touched bipartite graph.  Components with no observed edge
    are irrelevant to the reachable interface and are not counted in S.
    """
    left_words = np.asarray(left_words, dtype=np.int64)
    right_words = np.asarray(right_words, dtype=np.int64)
    if left_words.shape != right_words.shape:
        raise ValueError("left_words and right_words must have the same shape")
    n_side = int(q ** width)
    if len(left_words) == 0:
        # Vacuous reachable interface.  Report zero residual/cost rather than a
        # fake large S; callers should treat n_observed_pairs=0 as no data.
        return EqualizerSummary(
            q=q,
            width=width,
            alphabet_size_per_side=n_side,
            n_observed_pairs=0,
            edge_density=0.0,
            shared_classes=0,
            residual_qcoords=math.nan,
            residual_bits=math.nan,
            cost_qcoords=math.nan,
            cost_bits=math.nan,
            class_sizes_left=(),
            class_sizes_right=(),
            observed_left_words=0,
            observed_right_words=0,
        )

    pairs = np.unique(np.stack([left_words, right_words], axis=1), axis=0)
    uf = UF(2 * n_side)
    touched_left: set[int] = set()
    touched_right: set[int] = set()
    for l, r in pairs:
        l_i = int(l)
        r_i = int(r)
        if not (0 <= l_i < n_side and 0 <= r_i < n_side):
            raise ValueError("word outside alphabet range")
        touched_left.add(l_i)
        touched_right.add(r_i)
        uf.union(l_i, n_side + r_i)

    touched_roots = {uf.find(x) for x in touched_left} | {uf.find(n_side + x) for x in touched_right}
    shared = len(touched_roots)
    residual_q = math.log(shared, q) if shared > 0 else math.nan
    residual_b = math.log2(shared) if shared > 0 else math.nan
    cost_q = 2 * width - residual_q if shared > 0 else math.nan
    cost_b = 2 * width * math.log2(q) - residual_b if shared > 0 else math.nan

    # Class sizes on each side, restricted to touched words.  Useful to tell
    # diagonal-like laws from complete-collapse laws.
    left_sizes = []
    right_sizes = []
    for root in sorted(touched_roots):
        left_sizes.append(sum(1 for x in touched_left if uf.find(x) == root))
        right_sizes.append(sum(1 for x in touched_right if uf.find(n_side + x) == root))

    return EqualizerSummary(
        q=q,
        width=width,
        alphabet_size_per_side=n_side,
        n_observed_pairs=int(len(pairs)),
        edge_density=float(len(pairs) / (n_side * n_side)),
        shared_classes=int(shared),
        residual_qcoords=float(residual_q),
        residual_bits=float(residual_b),
        cost_qcoords=float(cost_q),
        cost_bits=float(cost_b),
        class_sizes_left=tuple(int(x) for x in left_sizes),
        class_sizes_right=tuple(int(x) for x in right_sizes),
        observed_left_words=int(len(touched_left)),
        observed_right_words=int(len(touched_right)),
    )


def _final_panel_words(ds: S.SCDCDiamond, maps: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finals = maps[:, states].reshape(-1)
    left_vertices = tuple(ds.offA + j for j in range(ds.w))
    right_vertices = tuple(ds.offA + ds.w + j for j in range(ds.w))
    left = _encode_word(_word_digits(finals, ds.q, left_vertices), ds.q)
    right = _encode_word(_word_digits(finals, ds.q, right_vertices), ds.q)
    return left, right


def _pairwise_equalizer(ds: S.SCDCDiamond, maps: np.ndarray, states: np.ndarray) -> dict:
    finals = maps[:, states].reshape(-1)
    summaries: list[EqualizerSummary] = []
    for j, (lv, rv) in enumerate(ds.panel_pairs):
        left = _digit(finals, ds.q, lv)
        right = _digit(finals, ds.q, rv)
        summaries.append(maximal_shared_equalizer(left, right, ds.q, 1))

    residual_q = float(sum(s.residual_qcoords for s in summaries))
    residual_b = float(sum(s.residual_bits for s in summaries))
    cost_q = float(2 * ds.w - residual_q)
    cost_b = float(2 * ds.w * math.log2(ds.q) - residual_b)
    shared_classes = [s.shared_classes for s in summaries]
    observed_pairs = [s.n_observed_pairs for s in summaries]
    densities = [s.edge_density for s in summaries]
    return {
        "eq_mode": "pairwise",
        "eq_shared_classes": json.dumps([int(x) for x in shared_classes]),
        "eq_shared_classes_mean": float(np.mean(shared_classes)) if shared_classes else math.nan,
        "eq_shared_classes_min": int(min(shared_classes)) if shared_classes else 0,
        "eq_n_observed_pairs": json.dumps([int(x) for x in observed_pairs]),
        "eq_edge_density_mean": float(np.mean(densities)) if densities else math.nan,
        "eq_residual_qcoords": residual_q,
        "eq_residual_bits": residual_b,
        "eq_cost_qcoords": cost_q,
        "eq_cost_bits": cost_b,
        "eq_residual_per_width": float(residual_q / ds.w) if ds.w else math.nan,
        "eq_cost_per_width": float(cost_q / ds.w) if ds.w else math.nan,
    }


def _joint_equalizer(ds: S.SCDCDiamond, maps: np.ndarray, states: np.ndarray) -> dict:
    left, right = _final_panel_words(ds, maps, states)
    s = maximal_shared_equalizer(left, right, ds.q, ds.w)
    out = s.as_dict(prefix="eq")
    out.update(
        eq_mode="joint",
        eq_shared_classes=json.dumps([int(s.shared_classes)]),
        eq_shared_classes_mean=float(s.shared_classes),
        eq_shared_classes_min=int(s.shared_classes),
        eq_n_observed_pairs=json.dumps([int(s.n_observed_pairs)]),
        eq_edge_density_mean=float(s.edge_density),
        eq_residual_per_width=float(s.residual_qcoords / ds.w) if ds.w else math.nan,
        eq_cost_per_width=float(s.cost_qcoords / ds.w) if ds.w else math.nan,
    )
    # Normalize names to match pairwise exactly.  as_dict uses eq_cost_qcoords etc already.
    return out


# --------------------------------------------------------------------------- #
# Main measurement
# --------------------------------------------------------------------------- #
def interface_equalizer_measure(
    ds: S.SCDCDiamond,
    eq_mode: EqualizerMode = "pairwise",
    initial_pool: InitialPool = "rec",
    observer_init: ObserverInit = "zero",
    compare_product: bool = True,
    precomputed_maps: tuple[np.ndarray, list[S.ScheduleInfo]] | None = None,
    precomputed_product: dict | None = None,
) -> dict:
    """Measure relational interface-equalizer cost on one diamond instance."""
    if eq_mode not in ("pairwise", "joint"):
        raise ValueError("eq_mode must be 'pairwise' or 'joint'")
    if precomputed_maps is None:
        maps, infos = S.step_maps(ds)
    else:
        maps, infos = precomputed_maps
    states = S._initial_states(ds, initial_pool, observer_init)  # internal but stable in patch series
    if eq_mode == "pairwise":
        out = _pairwise_equalizer(ds, maps, states)
    else:
        out = _joint_equalizer(ds, maps, states)

    out.update(
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

    # The equalizer is an interface relation, so by construction it imposes no
    # product-local quotient on A's hidden vertices.  Report explicit comparison
    # with the previous product-local sink_equal SCDC cost on the same instance.
    out["eq_hidden_A_cost_qcoords"] = 0.0
    out["eq_hidden_A_cost_bits"] = 0.0
    out["eq_panel_cost_qcoords"] = float(out["eq_cost_qcoords"])
    out["eq_panel_residual_qcoords"] = float(out["eq_residual_qcoords"])

    if compare_product:
        if precomputed_product is None:
            prod = S.scdc_diamond_quotient_cost(
                ds,
                constraint_mode="sink_equal",
                initial_pool=initial_pool,
                observer_init=observer_init,
                close_admissibility=True,
                precomputed_maps=(maps, infos),
            )
        else:
            prod = precomputed_product
        prod_panel = float(prod.get("cost_A_panel_qcoords", math.nan))
        prod_A = float(prod.get("cost_A_qcoords", math.nan))
        out.update(
            product_sink_cost_qcoords=float(prod["quotient_cost_qcoords"]),
            product_sink_cost_bits=float(prod["quotient_cost_bits"]),
            product_cost_A_qcoords=prod_A,
            product_cost_A_panel_qcoords=prod_panel,
            product_hidden_A_cost_qcoords=float(max(0.0, prod_A - prod_panel)) if not math.isnan(prod_A + prod_panel) else math.nan,
            product_collapsed_vertices=int(prod["collapsed_vertices"]),
            product_fully_collapsed_vertices=int(prod["fully_collapsed_vertices"]),
            product_admissibility_merges=int(prod["admissibility_merges"]),
        )
        out["overcollapse_avoided_qcoords"] = float(prod_A - out["eq_cost_qcoords"])
        out["hidden_collapse_avoided_qcoords"] = float(max(0.0, prod_A - prod_panel))
        out["equalizer_cost_fraction_of_product_A"] = float(out["eq_cost_qcoords"] / prod_A) if prod_A > 1e-12 else math.nan
    else:
        out.update(
            product_sink_cost_qcoords=math.nan,
            product_sink_cost_bits=math.nan,
            product_cost_A_qcoords=math.nan,
            product_cost_A_panel_qcoords=math.nan,
            product_hidden_A_cost_qcoords=math.nan,
            product_collapsed_vertices=0,
            product_fully_collapsed_vertices=0,
            product_admissibility_merges=0,
            overcollapse_avoided_qcoords=math.nan,
            hidden_collapse_avoided_qcoords=math.nan,
            equalizer_cost_fraction_of_product_A=math.nan,
        )
    return out


# --------------------------------------------------------------------------- #
# Sweeps / analysis / plotting / CLI
# --------------------------------------------------------------------------- #
def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _parse_modes(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def run_interface_equalizer_sweep(
    ks_B: Iterable[int] = (2,),
    ks_M: Iterable[int] = (2,),
    ks_A: Iterable[int] = (2, 3),
    ws: Iterable[int] = (1,),
    q: int = 4,
    n_instances: int = 20,
    eq_modes: Iterable[EqualizerMode] = ("pairwise", "joint"),
    initial_pool: InitialPool = "all",
    observer_init: ObserverInit = "zero",
    compare_product: bool = True,
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
                        seed = (base_seed * 1000033 + kB * 5113 + kM * 881 + kA * 137 + w * 23 + inst) % 2**32
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
                        maps_infos = S.step_maps(ds)
                        product_cache = None
                        if compare_product:
                            product_cache = S.scdc_diamond_quotient_cost(
                                ds,
                                constraint_mode="sink_equal",
                                initial_pool=initial_pool,
                                observer_init=observer_init,
                                close_admissibility=True,
                                precomputed_maps=maps_infos,
                            )
                        for mode in eq_modes:
                            m = interface_equalizer_measure(
                                ds,
                                eq_mode=mode,  # type: ignore[arg-type]
                                initial_pool=initial_pool,
                                observer_init=observer_init,
                                compare_product=compare_product,
                                precomputed_maps=maps_infos,
                                precomputed_product=product_cache,
                            )
                            m.update(seed=int(seed), instance=int(inst))
                            rows.append(m)
                    if verbose:
                        print(f"interface-equalizer kB={kB}, kM={kM}, kA={kA}, w={w} done", flush=True)
    return pd.DataFrame(rows)


def analyze_interface_equalizer(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"verdict": "NO DATA"}
    eps = 1e-9
    bound_viol = int((df["eq_cost_qcoords"] < -eps).sum())
    bound_viol += int((df["eq_residual_qcoords"] < -eps).sum())
    bound_viol += int((df["eq_residual_qcoords"] > df["w"].astype(float) + eps).sum())
    bound_viol += int((df["eq_cost_qcoords"] > 2 * df["w"].astype(float) + eps).sum())

    means = dict(
        mean_eq_residual_qcoords=float(df["eq_residual_qcoords"].mean()),
        mean_eq_cost_qcoords=float(df["eq_cost_qcoords"].mean()),
        mean_eq_residual_per_width=float(df["eq_residual_per_width"].mean()),
        mean_eq_cost_per_width=float(df["eq_cost_per_width"].mean()),
        mean_product_cost_A_qcoords=float(df["product_cost_A_qcoords"].mean()) if "product_cost_A_qcoords" in df else math.nan,
        mean_product_hidden_A_cost_qcoords=float(df["product_hidden_A_cost_qcoords"].mean()) if "product_hidden_A_cost_qcoords" in df else math.nan,
        mean_overcollapse_avoided_qcoords=float(df["overcollapse_avoided_qcoords"].mean()) if "overcollapse_avoided_qcoords" in df else math.nan,
        mean_hidden_collapse_avoided_qcoords=float(df["hidden_collapse_avoided_qcoords"].mean()) if "hidden_collapse_avoided_qcoords" in df else math.nan,
    )

    by_mode_df = (
        df.groupby("eq_mode")
        .agg(
            eq_residual_qcoords=("eq_residual_qcoords", "mean"),
            eq_cost_qcoords=("eq_cost_qcoords", "mean"),
            eq_residual_per_width=("eq_residual_per_width", "mean"),
            eq_cost_per_width=("eq_cost_per_width", "mean"),
            eq_shared_classes_mean=("eq_shared_classes_mean", "mean"),
            eq_edge_density_mean=("eq_edge_density_mean", "mean"),
            product_cost_A_qcoords=("product_cost_A_qcoords", "mean"),
            product_hidden_A_cost_qcoords=("product_hidden_A_cost_qcoords", "mean"),
            overcollapse_avoided_qcoords=("overcollapse_avoided_qcoords", "mean"),
        )
        .reset_index()
    )
    by_mode = {str(r.eq_mode): {k: float(getattr(r, k)) for k in by_mode_df.columns if k != "eq_mode"} for r in by_mode_df.itertuples(index=False)}

    scaling = []
    for keys, sub in df.groupby(["eq_mode", "kM", "kA", "w"]):
        eq_mode, kM, kA, w = keys
        by_k = sub.groupby("kB")["eq_cost_qcoords"].mean()
        by_k_resid = sub.groupby("kB")["eq_residual_qcoords"].mean()
        scaling.append(
            dict(
                eq_mode=str(eq_mode),
                kM=int(kM),
                kA=int(kA),
                w=int(w),
                equalizer_cost_by_kB={int(k): float(v) for k, v in by_k.items()},
                equalizer_residual_by_kB={int(k): float(v) for k, v in by_k_resid.items()},
            )
        )

    residual_ratio = means["mean_eq_residual_per_width"]
    avoided = means["mean_overcollapse_avoided_qcoords"]
    prod_hidden = means["mean_product_hidden_A_cost_qcoords"]
    if bound_viol:
        verdict = "IMPLEMENTATION WARNING: equalizer bounds violated"
    elif residual_ratio > 0.75 and avoided > 0.25:
        verdict = "INTERFACE EQUALIZER SIGNAL: shared boundary variable survives and product overcollapse is avoided"
    elif residual_ratio > 0.25 and avoided > 0.25:
        verdict = "PARTIAL INTERFACE EQUALIZER: nonzero shared boundary alphabet, but random panels are noisy/collapsed"
    elif prod_hidden > 0.25 and avoided > 0.25:
        verdict = "EQUALIZER AVOIDS HIDDEN-SINK COLLAPSE, but the shared interface alphabet is small in this regime"
    else:
        verdict = "NO NONTRIVIAL INTERFACE EQUALIZER in this regime"

    return dict(
        verdict=verdict,
        equalizer_bound_violations=int(bound_viol),
        **means,
        by_mode=by_mode,
        scaling=scaling,
    )


def plot_interface_equalizer(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("empty dataframe")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for mode in sorted(df.eq_mode.unique()):
        sub = df[df.eq_mode == mode]
        g = sub.groupby("w")["eq_residual_qcoords"].mean().dropna()
        ax[0].plot(g.index, g.values, "o-", label=str(mode))
    ax[0].set_xlabel("interface width w")
    ax[0].set_ylabel("residual shared interface (q-coords)")
    ax[0].set_title("equalizer residual")
    ax[0].legend(fontsize=8)

    for mode in sorted(df.eq_mode.unique()):
        sub = df[df.eq_mode == mode]
        g = sub.groupby("kA")["eq_cost_qcoords"].mean().dropna()
        ax[1].plot(g.index, g.values, "o-", label=f"eq {mode}")
    if "product_cost_A_qcoords" in df:
        g2 = df.groupby("kA")["product_cost_A_qcoords"].mean().dropna()
        ax[1].plot(g2.index, g2.values, "s--", label="product sink_equal")
    ax[1].set_xlabel("sink size kA")
    ax[1].set_ylabel("cost (q-coords)")
    ax[1].set_title("equalizer vs product quotient")
    ax[1].legend(fontsize=8)

    for mode in sorted(df.eq_mode.unique()):
        sub = df[df.eq_mode == mode]
        g = sub.groupby("kB")["eq_residual_qcoords"].mean().dropna()
        ax[2].plot(g.index, g.values, "o-", label=str(mode))
    ax[2].set_xlabel("source bulk kB")
    ax[2].set_ylabel("residual shared interface")
    ax[2].set_title("residual vs source bulk")
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
    ap.add_argument("--ks-A", default="2,3")
    ap.add_argument("--ws", default="1")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--eq-modes", default="pairwise,joint")
    ap.add_argument("--initial-pool", choices=("all", "rec"), default="all")
    ap.add_argument("--observer-init", choices=("zero", "all"), default="zero")
    ap.add_argument("--no-product", action="store_true", help="skip comparison to product-local SCDC sink_equal")
    ap.add_argument("--extra-B", type=int, default=None)
    ap.add_argument("--extra-M", type=int, default=None)
    ap.add_argument("--extra-A", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--max-joint-states", type=int, default=4 ** 9)
    ap.add_argument("--out", default="example_results/interface_equalizer.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df = run_interface_equalizer_sweep(
        ks_B=_parse_ints(args.ks_B),
        ks_M=_parse_ints(args.ks_M),
        ks_A=_parse_ints(args.ks_A),
        ws=_parse_ints(args.ws),
        q=args.q,
        n_instances=args.instances,
        eq_modes=_parse_modes(args.eq_modes),  # type: ignore[arg-type]
        initial_pool=args.initial_pool,  # type: ignore[arg-type]
        observer_init=args.observer_init,  # type: ignore[arg-type]
        compare_product=not args.no_product,
        extra_B=args.extra_B,
        extra_M=args.extra_M,
        extra_A=args.extra_A,
        base_seed=args.base_seed,
        max_joint_states=args.max_joint_states,
        verbose=not args.quiet,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    res = analyze_interface_equalizer(df)
    print(json.dumps(res, indent=2, default=float))
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    if args.plot:
        plot_interface_equalizer(df, args.plot)
        print(f"wrote {args.plot}")
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
