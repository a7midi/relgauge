"""stats.py -- bootstrap confidence intervals and simple diagnostics used by
the experiment report.  Everything here is standard, conservative statistics;
nothing theory-specific."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n_boot=2000, alpha=0.05, statistic=np.mean, rng=None):
    """Percentile bootstrap CI for a statistic of a 1-D sample."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = statistic(x[rng.integers(0, n, n)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(statistic(x)), float(lo), float(hi), int(n))


def trend_in(values_by_x):
    """Given {x: mean} for increasing x, report whether the sequence is
    monotone increasing / decreasing and the last-step change (a crude
    convergence indicator)."""
    xs = sorted(values_by_x)
    ys = [values_by_x[x] for x in xs]
    if len(ys) < 2:
        return dict(direction="n/a", last_delta=np.nan)
    diffs = np.diff(ys)
    if np.all(diffs > 0):
        d = "increasing"
    elif np.all(diffs < 0):
        d = "decreasing"
    else:
        d = "non-monotone"
    return dict(direction=d, last_delta=float(diffs[-1]),
                first=float(ys[0]), last=float(ys[-1]))


def q_independence(values_by_q, tol=0.03):
    """Is the q->value relation flat (within tol) vs drifting?  Also report the
    spread and whether it tracks the (1-1/q)^q image-artifact curve."""
    qs = sorted(values_by_q)
    ys = np.array([values_by_q[q] for q in qs], float)
    artifact = np.array([(1 - 1 / q) ** q for q in qs], float)
    spread = float(np.nanmax(ys) - np.nanmin(ys))
    # correlation with artifact curve (nan-safe)
    mask = ~np.isnan(ys)
    if mask.sum() >= 3 and np.std(ys[mask]) > 1e-9 and np.std(artifact[mask]) > 1e-9:
        corr = float(np.corrcoef(ys[mask], artifact[mask])[0, 1])
    else:
        corr = np.nan
    return dict(qs=qs, values=ys.tolist(), spread=spread,
                flat=bool(spread <= tol), corr_with_one_over_e=corr)
