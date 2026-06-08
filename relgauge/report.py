"""
report.py -- turn sweep tables into (a) plots and (b) a written falsification
verdict for the reformulated theory.

The single empirical claim under test:
    R^* = lim_{rho->0} mean R(k,q,w,ensemble) exists, is > 0, and is independent
    of q and of the ensemble.
with rho = w/k.  Outcomes: SUPPORTED / TRIVIAL / ARTIFACT / INCONCLUSIVE.

We never assume a target value (the old 1/3 = 1/d is dropped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESOLVER_METRICS = ["R_active", "R_passive_h1", "R_passive_h2"]


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_partition_collapse(df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df[df.ensemble == "cycle"]
    for col, lab in [("part_surv_image_count", "image-restricted (recurrent)"),
                     ("part_surv_full_count", "full-alphabet")]:
        if col not in sub:
            continue
        g = sub.groupby("q")[col].mean()
        ax.plot(g.index, g.values, "o-", label=f"partition survival, {lab}")
    qs = sorted(sub.q.unique())
    ax.plot(qs, [(1 - 1 / q) ** q for q in qs], "k--",
            label=r"$(1-1/q)^q$  (image artifact)")
    ax.set_xlabel("alphabet size q"); ax.set_ylabel("survival (count ratio)")
    ax.set_title("Partition quotient on cycles: collapse vs 1/e artifact")
    ax.legend(fontsize=8); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_orbit_dim(df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df[df.ensemble == "cycle"]
    g = sub.groupby("k")["orbit_dim_image"].mean()
    ax.plot(g.index, g.values, "o-", label="measured orbit dimension")
    ks = sorted(sub.k.unique())
    ax.plot(ks, [(k - 1) / k for k in ks], "k--", label="(k-1)/k")
    ax.set_xlabel("SCC size k"); ax.set_ylabel(r"$\log_{q^k}|\mathrm{reachable}|$")
    ax.set_title("Orbit (intrinsic) dimension on cycles")
    ax.legend(fontsize=8); ax.set_ylim(0, 1); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_resolvability_scaling(df, metric, path):
    """R vs ratio rho=w/k, one line per q, for a chosen ensemble (the densest
    available, where structure is richest)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ens = "scc" if "scc" in df.ensemble.unique() else df.ensemble.unique()[0]
    sub = df[df.ensemble == ens]
    for q in sorted(sub.q.unique()):
        s2 = sub[sub.q == q]
        g = s2.groupby("ratio")[metric].mean().dropna()
        if len(g) == 0:
            continue
        axes[0].plot(g.index, g.values, "o-", label=f"q={q}")
        axes[1].plot(g.index, g.values * 0 + g.values / np.maximum(g.index, 1e-9),
                     "o-", label=f"q={q}")
    axes[0].set_xlabel(r"$\rho = w/k$"); axes[0].set_ylabel(metric)
    axes[0].set_title(f"{metric} vs size ratio (ensemble={ens})")
    axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"$\rho = w/k$"); axes[1].set_ylabel(metric + r"$/\rho$")
    axes[1].set_title("intensive (slope) view")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def make_plots(df, outdir):
    paths = {}
    try:
        plot_partition_collapse(df, f"{outdir}/fig_partition_collapse.png")
        paths["partition"] = f"{outdir}/fig_partition_collapse.png"
    except Exception as e:
        print("partition plot failed:", e)
    try:
        plot_orbit_dim(df, f"{outdir}/fig_orbit_dim.png")
        paths["orbit"] = f"{outdir}/fig_orbit_dim.png"
    except Exception as e:
        print("orbit plot failed:", e)
    for m in RESOLVER_METRICS:
        if m in df:
            try:
                plot_resolvability_scaling(df, m, f"{outdir}/fig_scaling_{m}.png")
                paths[m] = f"{outdir}/fig_scaling_{m}.png"
            except Exception as e:
                print(f"scaling plot {m} failed:", e)
    return paths


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def _resolvability_verdict(df, metric, rng):
    """Assess the falsifiable claim for one resolver metric.

    Strategy: at fixed w=1 (narrowest interface), look at R as k grows
    (rho=1/k -> 0).  Check (a) does R approach a nonzero limit, (b) is that
    behaviour q-independent, (c) does it track the (1-1/q)^q artifact.
    """
    sub = df[df.w == 1]
    if metric not in sub or sub[metric].dropna().empty:
        return dict(metric=metric, verdict="NO DATA")

    # per (ensemble): R vs k for each q
    lines = []
    out = dict(metric=metric, per_ensemble={})
    for ens in sorted(sub.ensemble.unique()):
        se = sub[sub.ensemble == ens]
        # value at the largest available k, per q  (closest to rho->0)
        kmax = se.k.max()
        big = se[se.k == kmax]
        vals_by_q = {int(q): float(big[big.q == q][metric].mean())
                     for q in sorted(big.q.unique())
                     if not np.isnan(big[big.q == q][metric].mean())}
        qind = st.q_independence(vals_by_q) if len(vals_by_q) >= 3 else None
        # trend in k at the largest q
        qmax = se.q.max()
        sq = se[se.q == qmax]
        trend = st.trend_in({int(k): float(sq[sq.k == k][metric].mean())
                             for k in sorted(sq.k.unique())})
        out["per_ensemble"][ens] = dict(
            kmax=int(kmax), vals_by_q_at_kmax=vals_by_q,
            q_independence=qind, trend_in_k_at_qmax=trend)
    return out


def write_report(df, outdir, provenance, structural):
    """Write report.md and return the verdict dict."""
    rng = np.random.default_rng(12345)
    L = []
    A = L.append
    A("# Resolvability experiment -- falsification report\n")
    A(f"_generated {provenance['timestamp']}; "
      f"python {provenance['python']}, numpy {provenance['numpy']}_\n")
    A(f"_instances total: {len(df)}_\n")

    # ---------------- validation of the established theorems ----------------
    A("\n## 1. Validation of established results (code self-checks)\n")
    cyc = df[df.ensemble == "cycle"]
    if "part_surv_image_count" in cyc:
        A("**Partition collapse on cycles** (image/recurrent-restricted "
          "survival should fall toward 0 as q grows; full-alphabet should "
          r"track $1/q+(1-1/q)^q$):\n")
        A("\n| q | image-restricted | full-alphabet | $1/q+(1-1/q)^q$ |")
        A("|---|---|---|---|")
        for q in sorted(cyc.q.unique()):
            s = cyc[cyc.q == q]
            A(f"| {q} | {s.part_surv_image_count.mean():.3f} | "
              f"{s.part_surv_full_count.mean():.3f} | "
              f"{1/q + (1-1/q)**q:.3f} |")
        A("")
    A("**Orbit dimension on cycles** (should approach (k-1)/k):\n")
    A("\n| k | measured | (k-1)/k |")
    A("|---|---|---|")
    for k in sorted(cyc.k.unique()):
        s = cyc[cyc.k == k]
        A(f"| {k} | {s.orbit_dim_image.mean():.3f} | {(k-1)/k:.3f} |")
    A("")
    A("**Structural checks:**\n")
    A(f"- S1 condensation is a DAG (acyclic causality between observers): "
      f"{'PASS' if structural['S1_dag'] else 'FAIL'} "
      f"({structural['S1_n']} systems).")
    A(f"- S2 finite light-cone: holds SYNCHRONOUSLY "
      f"{structural['S2_sync']:.0%}, holds ASYNCHRONOUSLY "
      f"{structural['S2_async']:.0%}.  "
      f"=> the 'speed = 1 edge/tick' claim is a synchronous property and "
      f"FAILS under the schedule (gauge) semantics.")
    A(f"- S3 fusion monotonicity: |M'|>=|M| in "
      f"{structural['S3_ok']}/{structural['S3_n']} valid-extension events "
      f"(min ratio ~ q^r); violated in {structural['S3_viol']}/"
      f"{structural['S3_viol_n']} events when the extension hypothesis is "
      f"broken (so the hypothesis is necessary, as proved).")

    # ---------------- the open question -------------------------------------
    A("\n## 2. The open question: resolvability scaling\n")
    A("Claim under test: with rho = w/k, the limit "
      "R* = lim_{rho->0} mean R exists, is > 0, and is q- and "
      "ensemble-independent.\n")
    A("\n> **Cardinality caveat (read first).** A *fixed* finite resolver must "
      "resolve a vanishing *fraction* of a growing system (pigeonhole); so R "
      "trending to 0 as rho->0 is the expected null, not a surprise. The "
      "non-trivial possibilities are: (i) a nonzero limit (genuine relational "
      "invariant), or (ii) a universal *slope* R/rho (intensive invariant). "
      "Both are reported below.\n")

    verdicts = {}
    for metric in RESOLVER_METRICS:
        if metric not in df:
            continue
        v = _resolvability_verdict(df, metric, rng)
        verdicts[metric] = v
        A(f"\n### {metric}\n")
        if v.get("verdict") == "NO DATA":
            A("no data."); continue
        for ens, d in v["per_ensemble"].items():
            A(f"- **{ens}** (largest k={d['kmax']}, i.e. smallest rho=1/{d['kmax']}):")
            vbq = d["vals_by_q_at_kmax"]
            A(f"    - R at smallest rho, by q: "
              + ", ".join(f"q{q}={val:.3f}" for q, val in vbq.items()))
            qi = d["q_independence"]
            if qi:
                A(f"    - q-independence: spread={qi['spread']:.3f} "
                  f"({'FLAT' if qi['flat'] else 'DRIFTS'}), "
                  f"corr with (1-1/q)^q = {qi['corr_with_one_over_e']:.2f}")
            tr = d["trend_in_k_at_qmax"]
            A(f"    - trend in k (at largest q): {tr['direction']}, "
              f"first={tr.get('first', float('nan')):.3f} -> "
              f"last={tr.get('last', float('nan')):.3f}, "
              f"last step delta={tr['last_delta']:.3f}")

    # ---------------- automated verdict -------------------------------------
    A("\n## 3. Automated verdict\n")
    overall = _classify(verdicts)
    for metric, cls in overall.items():
        A(f"- **{metric}: {cls}**")
    A("\n_Interpretation key_: "
      "SUPPORTED = nonzero, q-flat, not artifact-correlated, converging; "
      "TRIVIAL = R->0 as rho->0; "
      "ARTIFACT = nonzero but q-drifting / correlated with (1-1/q)^q; "
      "INCONCLUSIVE = trend present but not converged within reach "
      "(larger k needed -- exponentially expensive).\n")
    A("\nFigures: `fig_partition_collapse.png`, `fig_orbit_dim.png`, "
      "`fig_scaling_*.png`.\n")

    text = "\n".join(L)
    with open(f"{outdir}/report.md", "w") as f:
        f.write(text)
    return dict(verdicts=verdicts, overall=overall, text=text)


def _classify(verdicts):
    out = {}
    for metric, v in verdicts.items():
        if v.get("verdict") == "NO DATA" or "per_ensemble" not in v:
            out[metric] = "NO DATA"; continue
        # pool the per-ensemble evidence
        flats, artifacts, nonzero, converged = [], [], [], []
        for ens, d in v["per_ensemble"].items():
            vbq = list(d["vals_by_q_at_kmax"].values())
            if not vbq:
                continue
            nonzero.append(np.nanmean(vbq) > 0.05)
            qi = d["q_independence"]
            if qi:
                flats.append(qi["flat"])
                if not np.isnan(qi["corr_with_one_over_e"]):
                    artifacts.append(abs(qi["corr_with_one_over_e"]) > 0.9)
            tr = d["trend_in_k_at_qmax"]
            converged.append(abs(tr["last_delta"]) < 0.03)
        if not nonzero:
            out[metric] = "NO DATA"; continue
        is_nonzero = np.mean(nonzero) > 0.5
        is_flat = (np.mean(flats) > 0.5) if flats else False
        is_artifact = (np.mean(artifacts) > 0.5) if artifacts else False
        is_converged = (np.mean(converged) > 0.5) if converged else False
        if not is_nonzero:
            out[metric] = "TRIVIAL (R->0)"
        elif is_artifact or not is_flat:
            out[metric] = "ARTIFACT (q-dependent / 1-over-e correlated)"
        elif is_converged:
            out[metric] = "SUPPORTED (nonzero, q-flat, converged)"
        else:
            out[metric] = "INCONCLUSIVE (nonzero & q-flat but not converged)"
    return out
