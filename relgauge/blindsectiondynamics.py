"""
blindsectiondynamics.py

Backward-compatible short alias for blindobserversectiondynamics.py.
Use either:

    python -m relgauge.blindobserversectiondynamics ...

or:

    python -m relgauge.blindsectiondynamics ...
"""
from __future__ import annotations

try:
    from . import blindobserversectiondynamics as _BOSD
    from . import blindobserverconnection as _BOC
except Exception:  # pragma: no cover
    import blindobserversectiondynamics as _BOSD  # type: ignore
    import blindobserverconnection as _BOC  # type: ignore

from .blindobserversectiondynamics import *  # noqa: F401,F403,E402

Candidate = _BOSD.SectionCandidate


def connection_consistency(summary):
    """Backward-compatible connection-only consistency component."""
    return _BOC.score_from_summary((summary or {}).get("bundle_summary", {}), cycle_objective="all")


def score_from_section_summary(summary, coherence_target=0.05):
    """Backward-compatible score name.

    ``coherence_target`` is accepted for old scripts; the new implementation uses
    an internal saturating coherent-state scale in section_consistency_score.
    """
    return _BOSD.section_consistency_score(summary)[0]


def evolve_section_dynamics(*args, coherence_target=0.05, **kwargs):
    """Backward-compatible evolution function name."""
    hist_df, winners, summary = _BOSD.evolve_observer_section_dynamics(*args, **kwargs)
    # Old scripts/tests expected these names.  The new names are more explicit.
    summary.setdefault("final_best_summary", summary.get("final_best_section_summary", {}))
    summary.setdefault("all_time_best_summary", summary.get("all_time_best_section_summary", {}))
    return hist_df, winners, summary


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_BOSD.main())
