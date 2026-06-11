"""
sharedsquarecertifier.py -- exact post-selection certificate for shared-square winners.

The blind shared-square search measures candidate squares on a sampled initial
ensemble (typically joint_random).  This certifier re-evaluates saved winners on
an exhaustive source domain and checks the structural reason that this is a true
certificate: in the shared-square topology, all non-source variables are updated
after their predecessors, so hidden initial values are overwritten by the
canonical feed-forward schedule.

It reports:
  * hidden-overwrite/topological certificate;
  * exact edge equalizer sizes on all source words;
  * exact source liveness and edge-transport scores;
  * path holonomy, generated group, and gauge covariance diagnostics;
  * optional randomized hidden perturbation check.

Recommended run
---------------
python -m relgauge.sharedsquarecertifier example_results/blind_shared_square_winners_q4_w1.pkl ^
  --top-n 50 ^
  --hidden-check-samples 256 ^
  --out example_results/shared_square_certifier_q4_w1.csv ^
  --plot example_results/fig_shared_square_certifier_q4_w1.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from . import sharedsquareholonomy as SH


def _safe_float(x, default=math.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_winners(path: str, top_n: int | None = None, require_square: bool = True):
    with open(path, "rb") as f:
        data = pickle.load(f)
    winners = data.get("winners", []) if isinstance(data, dict) else data
    out = []
    for i, item in enumerate(winners):
        if isinstance(item, dict) and "square" in item:
            sq = item["square"]
            metrics = dict(item.get("metrics", {}))
            fit = _safe_float(item.get("fitness", metrics.get("blind_fitness", 0.0)), 0.0)
        elif not require_square and hasattr(item, "joint"):
            sq = item
            metrics = {}
            fit = 0.0
        else:
            continue
        out.append((i, fit, sq, metrics))
    out.sort(key=lambda t: (
        bool(t[3].get("valid_square", False)),
        bool(t[3].get("nontrivial_path_holonomy", False)),
        float(t[3].get("min_residual_qcoords", 0.0) or 0.0),
        float(t[1] or 0.0),
    ), reverse=True)
    if top_n is not None and int(top_n) > 0:
        out = out[:int(top_n)]
    if not out:
        raise ValueError(f"no saved SharedSquare winners found in {path!r}")
    return out


def zero_pred_vertices(sq: SH.SharedSquare) -> tuple[int, ...]:
    return tuple(v for v in range(sq.joint.k) if len(sq.joint.preds[v]) == 0)


def schedule_respects_feedforward(sq: SH.SharedSquare) -> bool:
    pos = {int(v): i for i, v in enumerate(sq.schedule)}
    for v in range(sq.joint.k):
        for p in sq.joint.preds[v]:
            if int(p) not in pos or int(v) not in pos:
                return False
            if pos[int(p)] > pos[int(v)]:
                return False
    return True


def hidden_overwrite_certificate(sq: SH.SharedSquare) -> dict:
    z = zero_pred_vertices(sq)
    src = tuple(int(v) for v in sq.A_src)
    no_extra_sources = set(z) == set(src)
    feedforward = schedule_respects_feedforward(sq)
    return dict(
        zero_pred_vertices=json.dumps(list(z)),
        zero_pred_count=int(len(z)),
        source_count=int(len(src)),
        source_is_exactly_zero_pred=bool(no_extra_sources),
        schedule_respects_feedforward=bool(feedforward),
        hidden_overwrite_certified=bool(no_extra_sources and feedforward),
    )


def _source_all_states_with_hidden(sq: SH.SharedSquare, rng: np.random.Generator, n_hidden: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (base, hidden) states. base has all sources once with zeros elsewhere;
    hidden repeats source words while randomizing all non-source coordinates."""
    q = int(sq.q); w = int(sq.w); k = int(sq.joint.k)
    n_words = q ** w
    base = np.zeros((n_words, k), dtype=np.int64)
    for word in range(n_words):
        for j, v in enumerate(sq.A_src):
            base[word, int(v)] = (word // (q ** j)) % q
    if int(n_hidden) <= 0:
        return base, base.copy()
    reps = np.resize(np.arange(n_words, dtype=np.int64), max(n_words, int(n_hidden)))
    rng.shuffle(reps)
    hidden = rng.integers(0, q, size=(len(reps), k), dtype=np.int64)
    for j, v in enumerate(sq.A_src):
        hidden[:, int(v)] = (reps // (q ** j)) % q
    return base, hidden


def _encoded_outputs(sq: SH.SharedSquare, S1: np.ndarray) -> dict[str, np.ndarray]:
    return dict(
        B_l=SH._encode_words_from_state(S1, sq.B_left, sq.q),
        B_r=SH._encode_words_from_state(S1, sq.B_right, sq.q),
        C_l=SH._encode_words_from_state(S1, sq.C_left, sq.q),
        C_r=SH._encode_words_from_state(S1, sq.C_right, sq.q),
        DT_l=SH._encode_words_from_state(S1, sq.D_top_left, sq.q),
        DT_r=SH._encode_words_from_state(S1, sq.D_top_right, sq.q),
        DB_l=SH._encode_words_from_state(S1, sq.D_bottom_left, sq.q),
        DB_r=SH._encode_words_from_state(S1, sq.D_bottom_right, sq.q),
    )


def hidden_independence_check(sq: SH.SharedSquare, n_hidden: int, rng: np.random.Generator) -> dict:
    if int(n_hidden) <= 0:
        return dict(hidden_check_samples=0, hidden_output_independence=math.nan, hidden_output_mismatch_fraction=math.nan)
    base, hidden = _source_all_states_with_hidden(sq, rng, int(n_hidden))
    S1b = SH._step_states(sq.joint, base, sq.schedule)
    S1h = SH._step_states(sq.joint, hidden, sq.schedule)
    outb = _encoded_outputs(sq, S1b)
    outh = _encoded_outputs(sq, S1h)
    q = int(sq.q); w = int(sq.w)
    n_words = q ** w
    # Map hidden rows back to their source word and compare against base output.
    src_h = SH._encode_words_from_state(hidden, sq.A_src, sq.q)
    checks = []
    for key in outb:
        checks.append(outh[key] == outb[key][src_h])
    ok = np.logical_and.reduce(checks) if checks else np.ones(len(hidden), dtype=bool)
    return dict(
        hidden_check_samples=int(len(hidden)),
        hidden_output_independence=bool(np.all(ok)),
        hidden_output_mismatch_fraction=float(1.0 - np.mean(ok)) if len(ok) else math.nan,
    )


def certify_square(
    sq: SH.SharedSquare,
    rng: np.random.Generator | None = None,
    hidden_check_samples: int = 0,
    min_source: float = 0.80,
    min_transport: float = 0.80,
    min_accuracy: float = 0.95,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    cert = hidden_overwrite_certificate(sq)
    # Exhaustive over source words.  When hidden_overwrite_certified is true,
    # this is a full certificate over all hidden initial states by induction on
    # the feed-forward schedule.
    row = SH.measure_shared_square(
        sq,
        initial_mode="source_all",
        n_random_initial=max(1, int(sq.q) ** int(sq.w)),
        min_source=float(min_source),
        min_transport=float(min_transport),
        min_accuracy=float(min_accuracy),
        rng=rng,
    )
    q = int(sq.q)
    vals = [int(row.get(k, 0) or 0) for k in ["exact_B_classes", "exact_C_classes", "exact_Dtop_classes", "exact_Dbottom_classes"]]
    min_classes = min(vals) if vals else 0
    max_classes = max(vals) if vals else 0
    row.update(cert)
    row.update(
        certified_source_words=int(q ** int(sq.w)),
        certified_exhaustive_source=True,
        certified_min_classes=int(min_classes),
        certified_max_classes=int(max_classes),
        certified_min_residual_qcoords=float(math.log(min_classes, q)) if min_classes > 0 else 0.0,
        certified_valid_square=bool(row.get("valid_square", False) and cert["hidden_overwrite_certified"]),
    )
    row.update(hidden_independence_check(sq, hidden_check_samples, rng))
    return row


def run_certifier(
    winner_path: str,
    top_n: int = 50,
    hidden_check_samples: int = 0,
    base_seed: int = 0,
    min_source: float = 0.80,
    min_transport: float = 0.80,
    min_accuracy: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for rank, (idx, fit, sq, metrics) in enumerate(_load_winners(winner_path, top_n=top_n)):
        rng = np.random.default_rng((int(base_seed) * 1000003 + rank * 7919 + idx) % 2**32)
        r = certify_square(sq, rng, hidden_check_samples, min_source, min_transport, min_accuracy)
        r.update(
            winner_rank=int(rank),
            winner_index=int(idx),
            saved_fitness=float(fit),
            saved_valid_square=bool(metrics.get("valid_square", False)),
            saved_nontrivial_holonomy=bool(metrics.get("nontrivial_path_holonomy", False)),
            winner_path=str(winner_path),
        )
        rows.append(r)
    return pd.DataFrame(rows)


def analyze_certifier(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO CERTIFIED WINNERS", n_rows=0)
    def mean_bool(col):
        return float(df[col].fillna(False).astype(bool).mean()) if col in df else math.nan
    certified_frac = mean_bool("certified_valid_square")
    hidden_frac = mean_bool("hidden_overwrite_certified")
    nontriv_frac = mean_bool("nontrivial_path_holonomy") if "nontrivial_path_holonomy" in df else math.nan
    cov_frac = mean_bool("gauge_covariance_success") if "gauge_covariance_success" in df else math.nan
    c2_frac = float((df.get("generated_group_name", pd.Series([], dtype=str)).astype(str) == "C2").mean()) if "generated_group_name" in df else math.nan
    if certified_frac >= 0.95 and nontriv_frac > 0.0:
        verdict = "CERTIFIED SHARED-SQUARE HOLONOMY: saved winners remain exact under exhaustive source certification"
    elif certified_frac > 0.0:
        verdict = "PARTIAL SHARED-SQUARE CERTIFICATION: some saved winners certify exactly"
    else:
        verdict = "NO CERTIFIED SHARED-SQUARE WINNERS under this audit"
    return dict(
        verdict=verdict,
        n_rows=int(len(df)),
        certified_valid_fraction=float(certified_frac),
        hidden_overwrite_fraction=float(hidden_frac),
        nontrivial_holonomy_fraction=float(nontriv_frac),
        gauge_covariance_success_fraction=float(cov_frac),
        c2_fraction=float(c2_frac),
        group_counts={str(k): int(v) for k, v in Counter(df.get("generated_group_name", pd.Series([], dtype=str)).astype(str)).items()},
        delta_type_counts={str(k): int(v) for k, v in Counter(df.get("delta_type", pd.Series([], dtype=str)).astype(str)).items()},
        min_classes_counts={str(int(k)): int(v) for k, v in Counter(df.get("certified_min_classes", pd.Series([], dtype=int)).fillna(0).astype(int)).items()},
    )


def plot_certifier(df: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        for ax in axes: ax.set_axis_off()
    else:
        g = df.get("generated_group_name", pd.Series([], dtype=str)).astype(str).value_counts()
        axes[0].bar(g.index.astype(str), g.values)
        axes[0].set_title("certified generated group")
        axes[0].tick_params(axis="x", rotation=35)
        d = df.get("delta_type", pd.Series([], dtype=str)).astype(str).value_counts()
        axes[1].bar(d.index.astype(str), d.values)
        axes[1].set_title("certified holonomy type")
        axes[1].tick_params(axis="x", rotation=35)
        cls = df.get("certified_min_classes", pd.Series([], dtype=float)).dropna().astype(int).value_counts().sort_index()
        axes[2].bar([str(x) for x in cls.index], cls.values)
        axes[2].set_title("min exact classes")
        axes[2].set_xlabel("|S| min over edges")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Certify saved shared-square winners over the exact source domain.")
    ap.add_argument("winner_path")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--hidden-check-samples", type=int, default=0)
    ap.add_argument("--min-source", type=float, default=0.80)
    ap.add_argument("--min-transport", type=float, default=0.80)
    ap.add_argument("--min-accuracy", type=float, default=0.95)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", default="example_results/shared_square_certifier.csv")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)
    df = run_certifier(
        args.winner_path,
        top_n=args.top_n,
        hidden_check_samples=args.hidden_check_samples,
        base_seed=args.base_seed,
        min_source=args.min_source,
        min_transport=args.min_transport,
        min_accuracy=args.min_accuracy,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = analyze_certifier(df)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    if args.plot:
        plot_certifier(df, args.plot)
        print(f"wrote {args.plot}")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
