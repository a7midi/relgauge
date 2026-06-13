"""
observerconnectioncertifier.py

Exhaustive certifier for observer-frame connection winners.

This module is deliberately an audit layer.  It does not select, mutate, reward
C2, reward nontrivial holonomy, or add a physical structure.  It loads winners
saved by blindobserverconnection.py and re-runs observerframebundle.py with
exhaustive finite enumeration whenever the full state space is small enough.

For the controlled q=4, n=8 observer diamond, the full state space has 4^8 =
65536 states.  The certifier therefore enumerates all global frame-comparison
states and all local channel input/background assignments used by the edge
quotient extraction.  It then checks whether the winner really carries a true
multi-port observer-frame connection whose loop closes as a nontrivial C2
automorphism in local-chart coordinates.

CLI example
-----------
python -m relgauge.observerconnectioncertifier ^
  --winners example_results/blind_observer_connection_v4_q4_winners.pkl ^
  --winner-index 0 ^
  --frame-coordinate-mode local_charts ^
  --require-nontrivial ^
  --require-c2 ^
  --out example_results/certified_observer_connection_v4_q4.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # required for CLI output, optional at import time
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from . import blindobserverconnection as BOC
    from . import observerboundarygeometry as OBG
    from . import observerframebundle as OBF
except Exception:  # pragma: no cover
    import blindobserverconnection as BOC  # type: ignore
    import observerboundarygeometry as OBG  # type: ignore
    import observerframebundle as OBF  # type: ignore


def derived_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or ".csv")


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def load_winner_payload(path: str) -> List[Dict]:
    """Load a winners pickle produced by blindobserverconnection.py.

    The writer currently stores a list of dictionaries with keys
    ``system``, ``score``, ``summary``, and ``graph_id``.  This loader is kept
    tolerant of dataclass-like objects so old exploratory pickles remain usable.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "system" in payload:
        payload = [payload]
    if not isinstance(payload, list):
        raise TypeError(f"expected a list/dict winner payload, got {type(payload)!r}")
    out: List[Dict] = []
    for i, item in enumerate(payload):
        if isinstance(item, dict):
            if "system" not in item:
                raise KeyError(f"winner item {i} lacks 'system'")
            out.append(item)
        elif hasattr(item, "system"):
            out.append(dict(
                system=getattr(item, "system"),
                score=float(getattr(item, "score", 0.0)),
                summary=getattr(item, "summary", {}),
                graph_id=int(getattr(item, "graph_id", i)),
            ))
        else:
            raise TypeError(f"unsupported winner item {i}: {type(item)!r}")
    return out


def _all_indices(n: int, winner_index: int, all_winners: bool, best_only: bool, payload: List[Dict]) -> List[int]:
    if all_winners:
        return list(range(n))
    if best_only:
        scores = [float(item.get("score", item.get("summary", {}).get("all_time_best_score", 0.0))) for item in payload]
        return [int(max(range(n), key=lambda i: scores[i]))]
    if winner_index < 0:
        winner_index = n + int(winner_index)
    if not (0 <= winner_index < n):
        raise IndexError(f"winner_index {winner_index} outside 0..{n-1}")
    return [int(winner_index)]


def certification_checks(
    summary: Dict,
    exhaustive_states: bool,
    require_nontrivial: bool = True,
    require_c2: bool = True,
    require_nonflat: bool = True,
    path_agreement_eps: float = 1e-9,
) -> Tuple[List[Dict], bool, str]:
    """Return per-check rows, certified?, and verdict string."""
    checks: List[Dict] = []

    def add(name: str, passed: bool, value, expected: str, severity: str = "required") -> None:
        checks.append(dict(
            check=name,
            passed=int(bool(passed)),
            value=value,
            expected=expected,
            severity=severity,
        ))

    n_edges = int(summary.get("n_observer_edges", 0))
    n_frames = int(summary.get("n_frames", 0))
    n_trans = int(summary.get("n_frame_transports", 0))
    n_cycles = int(summary.get("n_observer_cycles", 0))
    nontriv = int(summary.get("n_nontrivial_holonomy", 0)) > 0 or int(summary.get("global_nontrivial_holonomy", 0)) > 0
    valid_auto = int(summary.get("n_loop_automorphism_valid", summary.get("n_valid_holonomy", 0))) > 0 or int(summary.get("global_valid_holonomy", 0)) > 0
    flat = int(summary.get("n_flat_connections", summary.get("n_valid_connections", 0))) > 0 or float(summary.get("global_flat_fraction", 0.0)) >= 1.0
    group = str(summary.get("global_generated_group", ""))

    add("exhaustive_state_space", exhaustive_states, exhaustive_states, "full q^n state enumeration used")
    add("full_system_boundary_empty", int(summary.get("full_boundary_violation", 1)) == 0, summary.get("full_boundary_violation"), "0")
    add("observer_edges_live", n_edges > 0 and int(summary.get("n_live_edge_quotients", 0)) == n_edges, f"{summary.get('n_live_edge_quotients', 0)}/{n_edges}", "all observer edges live")
    add("true_multi_port_frames", n_frames > 0 and int(summary.get("n_true_live_frames", 0)) == n_frames, f"{summary.get('n_true_live_frames', 0)}/{n_frames}", "all frames true multi-port")
    add("no_single_port_frame_leakage", int(summary.get("n_single_port_label_frames", 0)) == 0, summary.get("n_single_port_label_frames"), "0")
    add("true_frame_transports", n_trans > 0 and int(summary.get("n_true_live_frame_transports", 0)) == n_trans, f"{summary.get('n_true_live_frame_transports', 0)}/{n_trans}", "all transports true multi-port")
    add("usable_observer_diamond", int(summary.get("n_usable_diamonds", 0)) > 0, summary.get("n_usable_diamonds"), ">0")
    add("branch_completion", float(summary.get("max_branch_completion", 0.0)) >= 1.0 and float(summary.get("max_two_branch_completion", 0.0)) >= 1.0, f"{summary.get('max_branch_completion',0.0)}/{summary.get('max_two_branch_completion',0.0)}", "1.0/1.0")
    add("complete_branches", float(summary.get("max_complete_branch", 0.0)) >= 1.0 and float(summary.get("max_two_complete_branches", 0.0)) >= 1.0, f"{summary.get('max_complete_branch',0.0)}/{summary.get('max_two_complete_branches',0.0)}", "1.0/1.0")
    add("path_comparability", float(summary.get("max_path_comparability", 0.0)) >= 1.0 and float(summary.get("max_path_bijective", 0.0)) >= 1.0, f"{summary.get('max_path_comparability',0.0)}/{summary.get('max_path_bijective',0.0)}", "1.0/1.0")
    add("loop_automorphism_valid", valid_auto, summary.get("n_loop_automorphism_valid", summary.get("n_valid_holonomy", 0)), ">0")
    add("global_holonomy_valid", int(summary.get("global_valid_holonomy", 0)) > 0, summary.get("global_valid_holonomy"), ">0")
    add("nontrivial_holonomy", (nontriv if require_nontrivial else True), summary.get("global_nontrivial_holonomy", summary.get("n_nontrivial_holonomy", 0)), ">0" if require_nontrivial else "not required")
    add("nonflat_path_mismatch", (float(summary.get("max_path_agreement", 1.0)) < 1.0 - path_agreement_eps if require_nonflat else True), summary.get("max_path_agreement"), "<1" if require_nonflat else "not required")
    add("generated_group_c2", (group == "C2" if require_c2 else True), group, "C2" if require_c2 else "not required")
    add("binary_frames", abs(float(summary.get("mean_frame_classes", 0.0)) - 2.0) < 1e-9, summary.get("mean_frame_classes"), "2.0", severity="recommended")
    add("local_chart_mode", str(summary.get("frame_coordinate_mode", "")) == "local_charts", summary.get("frame_coordinate_mode"), "local_charts")

    required_ok = all(bool(r["passed"]) for r in checks if r.get("severity") == "required")
    if required_ok and require_nontrivial and require_c2:
        verdict = "CERTIFIED NON-FLAT C2 OBSERVER-FRAME CONNECTION"
    elif required_ok and require_nontrivial:
        verdict = "CERTIFIED NON-FLAT OBSERVER-FRAME CONNECTION"
    elif required_ok:
        verdict = "CERTIFIED OBSERVER-FRAME CONNECTION"
    else:
        verdict = "CERTIFICATION FAILED: at least one required exhaustive check failed"
    return checks, bool(required_ok), verdict


def certify_system(
    sys: OBG.FiniteRelationalSystem,
    graph_id: int = 0,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    cycle_max_len: int = 6,
    state_warmup: int = 0,
    max_exhaustive_states: int = 1_000_000,
    require_nontrivial: bool = True,
    require_c2: bool = True,
    require_nonflat: bool = True,
    seed: int = 0,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], Dict, List[Dict]]:
    """Run exhaustive observer-frame analysis and attach certification checks."""
    q = int(sys.q)
    k = int(sys.k)
    full_states = int(q ** k)
    exhaustive_states = bool(full_states <= int(max_exhaustive_states))
    if not exhaustive_states:
        raise ValueError(
            f"full state space q^n={full_states} exceeds --max-exhaustive-states={max_exhaustive_states}; "
            "increase the cap only if this is intentional"
        )
    rng = np.random.default_rng(seed)
    # For the local channel audits, using q^n as a cap is conservative: every
    # input/background domain used by the local edge quotient extractor is no
    # larger than the full finite state space in these systems.
    max_exact = int(full_states)
    fr, er, tr, cr, summary = OBF.analyze_system_bundle(
        sys,
        graph_id=graph_id,
        max_channel_inputs=max_exact,
        max_channel_backgrounds=max_exact,
        max_state_samples=max_exact,
        state_warmup=state_warmup,
        cycle_max_len=cycle_max_len,
        min_frame_ports=min_frame_ports,
        frame_coordinate_mode=frame_coordinate_mode,
        rng=rng,
    )
    score = BOC.score_from_summary(summary)
    summary = dict(summary)
    summary.update(
        exhaustive_state_space=True,
        exhaustive_state_count=full_states,
        exhaustive_channel_cap=max_exact,
        certification_score=float(score),
        certification_state_warmup=int(state_warmup),
    )
    checks, certified, verdict = certification_checks(
        summary,
        exhaustive_states=True,
        require_nontrivial=require_nontrivial,
        require_c2=require_c2,
        require_nonflat=require_nonflat,
    )
    summary.update(certified=bool(certified), certification_verdict=verdict)
    return fr, er, tr, cr, summary, checks


def certify_winners(
    winners_path: str,
    winner_index: int = 0,
    all_winners: bool = False,
    best_only: bool = False,
    min_frame_ports: int = 2,
    frame_coordinate_mode: str = "local_charts",
    cycle_max_len: int = 6,
    state_warmup: int = 0,
    max_exhaustive_states: int = 1_000_000,
    require_nontrivial: bool = True,
    require_c2: bool = True,
    require_nonflat: bool = True,
    seed: int = 0,
) -> Tuple[object, object, object, object, object, Dict]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for certification output")
    payload = load_winner_payload(winners_path)
    indices = _all_indices(len(payload), winner_index, all_winners, best_only, payload)
    all_frames: List[Dict] = []
    all_edges: List[Dict] = []
    all_trans: List[Dict] = []
    all_cycles: List[Dict] = []
    all_checks: List[Dict] = []
    summary_rows: List[Dict] = []

    for out_i, idx in enumerate(indices):
        item = payload[idx]
        sys = item["system"]
        fr, er, tr, cr, summary, checks = certify_system(
            sys,
            graph_id=int(item.get("graph_id", idx)),
            min_frame_ports=min_frame_ports,
            frame_coordinate_mode=frame_coordinate_mode,
            cycle_max_len=cycle_max_len,
            state_warmup=state_warmup,
            max_exhaustive_states=max_exhaustive_states,
            require_nontrivial=require_nontrivial,
            require_c2=require_c2,
            require_nonflat=require_nonflat,
            seed=seed + out_i,
        )
        base = dict(
            winner_index=int(idx),
            input_score=float(item.get("score", 0.0)),
            input_graph_id=int(item.get("graph_id", idx)),
        )
        for coll in (fr, er, tr, cr, checks):
            for row in coll:
                row.update(base)
        srow = dict(base)
        srow.update(summary)
        summary_rows.append(_jsonable(srow))
        all_frames.extend(fr); all_edges.extend(er); all_trans.extend(tr); all_cycles.extend(cr); all_checks.extend(checks)

    frame_df = pd.DataFrame(all_frames)
    edge_df = pd.DataFrame(all_edges)
    trans_df = pd.DataFrame(all_trans)
    cycle_df = pd.DataFrame(all_cycles)
    check_df = pd.DataFrame(all_checks)
    summary_df = pd.DataFrame(summary_rows)

    certified_count = int(summary_df["certified"].sum()) if len(summary_df) and "certified" in summary_df else 0
    verdict_counts = {str(k): int(v) for k, v in Counter(summary_df.get("certification_verdict", [])).items()}
    group_counts = {str(k): int(v) for k, v in Counter(summary_df.get("global_generated_group", [])).items()}
    aggregate = dict(
        verdict=(
            "EXHAUSTIVE CERTIFICATION PASSED FOR ALL SELECTED WINNERS"
            if certified_count == len(summary_df) and len(summary_df) > 0
            else "EXHAUSTIVE CERTIFICATION INCOMPLETE/FAILED FOR SOME SELECTED WINNERS"
        ),
        winners_path=winners_path,
        n_payload_winners=int(len(payload)),
        n_certified_requested=int(len(indices)),
        n_certified_passed=certified_count,
        certified_fraction=float(certified_count / len(summary_df)) if len(summary_df) else 0.0,
        require_nontrivial=bool(require_nontrivial),
        require_c2=bool(require_c2),
        require_nonflat=bool(require_nonflat),
        min_frame_ports=int(min_frame_ports),
        frame_coordinate_mode=str(frame_coordinate_mode),
        state_warmup=int(state_warmup),
        verdict_counts=verdict_counts,
        group_counts=group_counts,
        mean_certification_score=float(summary_df["certification_score"].mean()) if len(summary_df) and "certification_score" in summary_df else 0.0,
        mean_path_agreement=float(summary_df["max_path_agreement"].mean()) if len(summary_df) and "max_path_agreement" in summary_df else 0.0,
        total_global_nontrivial_holonomy=int(summary_df["global_nontrivial_holonomy"].sum()) if len(summary_df) and "global_nontrivial_holonomy" in summary_df else 0,
    )
    return frame_df, edge_df, trans_df, cycle_df, check_df, dict(summary_rows=summary_rows, aggregate=_jsonable(aggregate))


def write_outputs(frame_df, edge_df, trans_df, cycle_df, check_df, result: Dict, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Main out is the per-winner summary table for easy spreadsheet use.
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas required")
    pd.DataFrame(result["summary_rows"]).to_csv(out, index=False)
    frame_df.to_csv(derived_path(out, "_frames"), index=False)
    edge_df.to_csv(derived_path(out, "_edges"), index=False)
    trans_df.to_csv(derived_path(out, "_transports"), index=False)
    cycle_df.to_csv(derived_path(out, "_cycles"), index=False)
    check_df.to_csv(derived_path(out, "_checks"), index=False)
    with open(derived_path(out, "_summary").replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exhaustively certify observer-frame connection winners")
    p.add_argument("--winners", required=True, help="pickle saved by blindobserverconnection.py")
    p.add_argument("--winner-index", type=int, default=0, help="winner index to certify; ignored by --all or --best-only")
    p.add_argument("--all", action="store_true", help="certify every stored winner payload item")
    p.add_argument("--best-only", action="store_true", help="certify the highest-score stored winner")
    p.add_argument("--min-frame-ports", type=int, default=2)
    p.add_argument("--frame-coordinate-mode", choices=["local_charts", "common_factor"], default="local_charts")
    p.add_argument("--cycle-max-len", type=int, default=6)
    p.add_argument("--state-warmup", type=int, default=0, help="0 gives all global states; use >0 only for recurrent/warm-state certification")
    p.add_argument("--max-exhaustive-states", type=int, default=1_000_000)
    p.add_argument("--require-nontrivial", action="store_true", default=False, help="require nontrivial holonomy for pass")
    p.add_argument("--allow-flat", action="store_true", help="do not require path mismatch/non-flatness")
    p.add_argument("--require-c2", action="store_true", default=False, help="require post-hoc global generated group C2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="example_results/certified_observer_connection.csv")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    frame_df, edge_df, trans_df, cycle_df, check_df, result = certify_winners(
        winners_path=args.winners,
        winner_index=args.winner_index,
        all_winners=bool(args.all),
        best_only=bool(args.best_only),
        min_frame_ports=args.min_frame_ports,
        frame_coordinate_mode=args.frame_coordinate_mode,
        cycle_max_len=args.cycle_max_len,
        state_warmup=args.state_warmup,
        max_exhaustive_states=args.max_exhaustive_states,
        require_nontrivial=bool(args.require_nontrivial),
        require_c2=bool(args.require_c2),
        require_nonflat=not bool(args.allow_flat),
        seed=args.seed,
    )
    write_outputs(frame_df, edge_df, trans_df, cycle_df, check_df, result, args.out)
    print(json.dumps(_jsonable(result["aggregate"]), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {derived_path(args.out, '_checks')}")
    print(f"wrote {derived_path(args.out, '_summary').replace('.csv', '.json')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
