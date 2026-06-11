"""
latticefusionaudit.py -- fuse independently selected Z2 lattice vacua and audit seam defects.

This is an Endpoint-D diagnostic that does *not* add matter.  It loads saved
microscopic Z2 lattice winners, places two independently selected vacua next to
one another, identifies the shared boundary, and asks whether the combined
system remains a valid transported C2/Z2 lattice or leaves a localized seam
residue.

The experiment is intentionally diagnostic:

  * no target defect type is inserted;
  * no flux entropy, excitation lifetime, or matter fitness is used by default;
  * the old lattice interiors are copied from selected winners;
  * the shared boundary is forced by identifying the two boundary vertex sets;
  * seam defects are measured as residual invalid/non-Z2 plaquettes, flux
    mismatches, and Wilson-loop response near the interface.

If two selected vacua cannot be fused without a localized residue, that residue
is a natural candidate for a matter-like defect: a consistency obstruction
between independently selected gauge vacua.

Recommended run
---------------
python -m relgauge.latticefusionaudit \
  example_results/lattice_growth_winners_q4_3to4.pkl ^
  --top-n 20 ^
  --pair-samples 80 ^
  --orientation horizontal ^
  --initial-mode joint_random ^
  --n-random-initial 4096 ^
  --out example_results/lattice_fusion_q4_4x4.csv ^
  --plot example_results/fig_lattice_fusion_q4_4x4.png

Optional seam-repair probe, still without specifying a defect type:

python -m relgauge.latticefusionaudit \
  example_results/lattice_growth_winners_q4_3to4.pkl ^
  --top-n 20 --pair-samples 40 --repair-generations 80 --population 24 ^
  --out example_results/lattice_fusion_repair_q4_4x4.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
from collections import Counter
from dataclasses import replace
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import core as C
from . import microlattice as ML
from . import blindmicrolattice as BM
from .sharedsquareholonomy import _mutate_rule

Orientation = Literal["horizontal", "vertical"]
InitialMode = Literal["source_all", "joint_random"]


# --------------------------------------------------------------------------- #
# Winner loading
# --------------------------------------------------------------------------- #
def load_lattice_winners(path: str, top_n: int | None = None) -> list[dict]:
    """Load winner records containing ``MicroLattice`` objects.

    Accepted formats are the ``dict(kind=..., winners=[...])`` files written by
    blindmicrolattice/latticegrowth, or a raw list of winner records.  Records
    are sorted by lattice validity and fitness before truncation.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "winners" in obj:
        winners = list(obj["winners"])
    elif isinstance(obj, list):
        winners = list(obj)
    else:
        raise ValueError(f"unrecognized winner file format: {path!r}")
    out: list[dict] = []
    for w in winners:
        if isinstance(w, dict) and "lattice" in w and hasattr(w["lattice"], "nx"):
            out.append(w)
    out.sort(key=lambda r: (
        bool(r.get("metrics", {}).get("full_grown_lattice", False)),
        bool(r.get("metrics", {}).get("full_valid_lattice", False)),
        float(r.get("metrics", {}).get("z2_plaquette_fraction", 0.0)),
        float(r.get("metrics", {}).get("growth_z2_fraction", 0.0)),
        float(r.get("fitness", r.get("metrics", {}).get("blind_fitness", 0.0))),
    ), reverse=True)
    if top_n is not None and int(top_n) > 0:
        out = out[:int(top_n)]
    return out


# --------------------------------------------------------------------------- #
# Semantic copying into a fused lattice
# --------------------------------------------------------------------------- #
def _edge_key(src: tuple[int, int], dst: tuple[int, int]):
    return (tuple(src), tuple(dst))


def _copy_rule_if_compatible(src: ML.MicroLattice, dst: ML.MicroLattice, src_v: int, dst_v: int) -> bool:
    if int(src.q) != int(dst.q):
        return False
    if len(src.joint.preds[int(src_v)]) != len(dst.joint.preds[int(dst_v)]):
        return False
    r1 = np.asarray(src.joint.rules[int(src_v)], dtype=np.int64)
    r2 = np.asarray(dst.joint.rules[int(dst_v)], dtype=np.int64)
    if r1.shape != r2.shape:
        return False
    dst.joint.rules[int(dst_v)] = r1.copy()
    return True


def _shift_coord(coord: tuple[int, int], orientation: Orientation, offset: int) -> tuple[int, int]:
    r, c = int(coord[0]), int(coord[1])
    if orientation == "horizontal":
        return (r, c + int(offset))
    if orientation == "vertical":
        return (r + int(offset), c)
    raise ValueError(f"unknown orientation {orientation!r}")


def _copy_lattice_region(
    src: ML.MicroLattice,
    dst: ML.MicroLattice,
    orientation: Orientation,
    offset: int,
    occupied: set[int] | None = None,
    tag: str = "region",
) -> dict:
    """Copy a source lattice into a shifted region of a destination lattice.

    ``occupied`` records destination vertices already copied from another
    region.  If a shifted source vertex lands on an occupied destination vertex
    (the fused boundary), the old rule is preserved and the conflict is counted.
    This implements boundary identification: the two vacua must share one
    boundary, not overwrite each other's boundary rules.
    """
    if occupied is None:
        occupied = set()
    copied: set[int] = set()
    skipped_overlap = 0
    incompatible = 0
    attempts = 0

    def try_copy(src_v: int, dst_v: int) -> None:
        nonlocal skipped_overlap, incompatible, attempts
        attempts += 1
        dst_v = int(dst_v); src_v = int(src_v)
        if dst_v in occupied:
            skipped_overlap += 1
            return
        if _copy_rule_if_compatible(src, dst, src_v, dst_v):
            occupied.add(dst_v); copied.add(dst_v)
        else:
            incompatible += 1

    # Corner outputs.
    for coord, src_vs in src.node_out.items():
        dcoord = _shift_coord(coord, orientation, offset)
        if dcoord not in dst.node_out:
            incompatible += len(src_vs)
            continue
        dst_vs = dst.node_out[dcoord]
        for sv, dv in zip(src_vs, dst_vs):
            try_copy(int(sv), int(dv))

    # Edge transporter vertices.
    for key, src_ed in src.edges.items():
        s0, s1 = key
        dkey = _edge_key(_shift_coord(s0, orientation, offset), _shift_coord(s1, orientation, offset))
        if dkey not in dst.edges:
            # Edge lies outside destination; should not happen for valid shifts.
            for attr in ("branch_left", "branch_right", "panel_left", "panel_right"):
                incompatible += len(getattr(src_ed, attr))
            continue
        dst_ed = dst.edges[dkey]
        for attr in ("branch_left", "branch_right", "panel_left", "panel_right"):
            for sv, dv in zip(getattr(src_ed, attr), getattr(dst_ed, attr)):
                try_copy(int(sv), int(dv))
        try:
            dst_ed.flip = int(src_ed.flip)
        except Exception:
            pass
    return dict(
        tag=str(tag), attempts=int(attempts), copied=int(len(copied)),
        skipped_overlap=int(skipped_overlap), incompatible=int(incompatible),
        copied_vertices=copied,
    )


def fuse_lattices(
    left: ML.MicroLattice,
    right: ML.MicroLattice,
    orientation: Orientation = "horizontal",
    rng: np.random.Generator | None = None,
    background_ensemble: str = "random",
) -> tuple[ML.MicroLattice, dict]:
    """Construct a fused lattice from two selected vacua.

    Horizontal fusion identifies the right boundary of ``left`` with the left
    boundary of ``right``.  Vertical fusion identifies the bottom boundary of
    the first lattice with the top boundary of the second.  Boundary overlaps
    are intentionally not overwritten by the second lattice: the seam is where
    compatibility has to be measured.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if int(left.q) != int(right.q) or int(left.w) != int(right.w):
        raise ValueError("lattices must have the same q and w")
    if orientation == "horizontal":
        if int(left.ny) != int(right.ny):
            raise ValueError("horizontal fusion requires equal ny")
        nx, ny = int(left.nx) + int(right.nx), int(left.ny)
        offset = int(left.nx)
    elif orientation == "vertical":
        if int(left.nx) != int(right.nx):
            raise ValueError("vertical fusion requires equal nx")
        nx, ny = int(left.nx), int(left.ny) + int(right.ny)
        offset = int(left.ny)
    else:
        raise ValueError(f"unknown orientation {orientation!r}")

    fused = ML.make_microscopic_lattice(q=int(left.q), nx=nx, ny=ny, w=int(left.w), rng=rng, ensemble=background_ensemble)
    occupied: set[int] = set()
    c1 = _copy_lattice_region(left, fused, orientation, 0, occupied=occupied, tag="left/top")
    c2 = _copy_lattice_region(right, fused, orientation, offset, occupied=occupied, tag="right/bottom")
    fused.meta = dict(fused.meta)
    fused.meta.update(kind="fused_lattice", orientation=str(orientation), left_nx=int(left.nx), left_ny=int(left.ny), right_nx=int(right.nx), right_ny=int(right.ny), background_ensemble=str(background_ensemble))
    meta = dict(copy_left=c1, copy_right=c2, orientation=str(orientation), offset=int(offset), nx=int(nx), ny=int(ny))
    return fused, meta


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _frac(rows: list[dict], pred) -> float:
    return float(sum(1 for x in rows if pred(x)) / max(1, len(rows)))


def _is_z2(p: dict) -> bool:
    try:
        return bool(p.get("valid", False)) and math.isfinite(float(p.get("flux", math.nan)))
    except Exception:
        return False


def _is_nontrivial(p: dict) -> bool:
    return bool(p.get("valid", False)) and bool(p.get("nontrivial", False))


def _region_rows(meas: dict, orientation: Orientation, offset: int) -> dict[str, list[dict]]:
    rows = list(meas.get("plaquettes", []))
    if orientation == "horizontal":
        left = [p for p in rows if int(p.get("c", -1)) < int(offset)]
        right = [p for p in rows if int(p.get("c", -1)) >= int(offset)]
        seam = [p for p in rows if int(p.get("c", -999)) in (int(offset)-1, int(offset))]
        interior = [p for p in rows if int(p.get("c", -1)) < int(offset)-1 or int(p.get("c", -1)) > int(offset)]
    else:
        left = [p for p in rows if int(p.get("r", -1)) < int(offset)]
        right = [p for p in rows if int(p.get("r", -1)) >= int(offset)]
        seam = [p for p in rows if int(p.get("r", -999)) in (int(offset)-1, int(offset))]
        interior = [p for p in rows if int(p.get("r", -1)) < int(offset)-1 or int(p.get("r", -1)) > int(offset)]
    return dict(left=left, right=right, seam=seam, interior=interior, all=rows)


def _region_metrics(prefix: str, rows: list[dict]) -> dict:
    return {
        f"{prefix}_plaquette_count": int(len(rows)),
        f"{prefix}_z2_fraction": _frac(rows, _is_z2),
        f"{prefix}_valid_fraction": _frac(rows, lambda p: bool(p.get("valid", False))),
        f"{prefix}_nontrivial_fraction": _frac([p for p in rows if bool(p.get("valid", False))], _is_nontrivial) if rows else 0.0,
    }


def _seam_mismatch(meas: dict, orientation: Orientation, offset: int) -> dict:
    F = np.asarray(meas.get("flux_grid"), dtype=float)
    if F.ndim != 2 or F.size == 0:
        return dict(seam_pair_count=0, seam_pair_z2_fraction=0.0, seam_flux_mismatch_fraction=math.nan)
    mismatches = []
    finite_pairs = 0
    if orientation == "horizontal":
        cL, cR = int(offset) - 1, int(offset)
        if cL < 0 or cR >= F.shape[1]:
            return dict(seam_pair_count=0, seam_pair_z2_fraction=0.0, seam_flux_mismatch_fraction=math.nan)
        for r in range(F.shape[0]):
            a, b = F[r, cL], F[r, cR]
            if np.isfinite(a) and np.isfinite(b):
                finite_pairs += 1; mismatches.append(float(a != b))
    else:
        rT, rB = int(offset) - 1, int(offset)
        if rT < 0 or rB >= F.shape[0]:
            return dict(seam_pair_count=0, seam_pair_z2_fraction=0.0, seam_flux_mismatch_fraction=math.nan)
        for c in range(F.shape[1]):
            a, b = F[rT, c], F[rB, c]
            if np.isfinite(a) and np.isfinite(b):
                finite_pairs += 1; mismatches.append(float(a != b))
    total_pairs = F.shape[0] if orientation == "horizontal" else F.shape[1]
    return dict(
        seam_pair_count=int(total_pairs),
        seam_pair_z2_fraction=float(finite_pairs / max(1, total_pairs)),
        seam_flux_mismatch_fraction=float(np.mean(mismatches)) if mismatches else math.nan,
    )


def _wilson_crossing_summary(meas: dict, orientation: Orientation, offset: int, max_loop_side: int = 4) -> dict:
    F = np.asarray(meas.get("flux_grid"), dtype=float)
    if F.ndim != 2 or F.size == 0:
        return dict(wilson_crossing_n=0, wilson_noncrossing_n=0, wilson_crossing_abs=math.nan, wilson_noncrossing_abs=math.nan, wilson_crossing_gap=math.nan)
    ny, nx = F.shape
    cross_vals: list[float] = []
    non_vals: list[float] = []
    for a in range(1, min(int(max_loop_side), nx) + 1):
        for b in range(1, min(int(max_loop_side), ny) + 1):
            for r in range(0, ny - b + 1):
                for c in range(0, nx - a + 1):
                    sub = F[r:r+b, c:c+a]
                    if not np.all(np.isfinite(sub)):
                        continue
                    val = float(np.prod(sub))
                    if orientation == "horizontal":
                        crosses = (c < int(offset) <= c + a - 1)
                    else:
                        crosses = (r < int(offset) <= r + b - 1)
                    (cross_vals if crosses else non_vals).append(val)
    ca = float(abs(np.mean(cross_vals))) if cross_vals else math.nan
    na = float(abs(np.mean(non_vals))) if non_vals else math.nan
    return dict(
        wilson_crossing_n=int(len(cross_vals)),
        wilson_noncrossing_n=int(len(non_vals)),
        wilson_crossing_abs=ca,
        wilson_noncrossing_abs=na,
        wilson_crossing_gap=float(ca - na) if math.isfinite(ca) and math.isfinite(na) else math.nan,
    )


def evaluate_fusion(
    left: ML.MicroLattice,
    right: ML.MicroLattice,
    orientation: Orientation = "horizontal",
    rng: np.random.Generator | None = None,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    max_loop_side: int = 4,
    background_ensemble: str = "random",
) -> tuple[dict, ML.MicroLattice]:
    if rng is None:
        rng = np.random.default_rng(0)
    fused, meta = fuse_lattices(left, right, orientation=orientation, rng=rng, background_ensemble=background_ensemble)
    meas = ML.measure_microscopic_lattice(fused, initial_mode=initial_mode, n_random_initial=int(n_random_initial), rng=rng)
    offset = int(meta["offset"])
    regions = _region_rows(meas, orientation, offset)
    row: dict = dict(
        q=int(fused.q), w=int(fused.w), nx=int(fused.nx), ny=int(fused.ny), orientation=str(orientation), offset=int(offset),
        z2_fraction=float(meas.get("z2_fraction", 0.0)),
        valid_fraction=float(meas.get("valid_fraction", 0.0)),
        nontrivial_fraction=float(meas.get("nontrivial_fraction", 0.0)),
        group_name_counts=json.dumps(meas.get("group_counts", {})),
        delta_type_counts=json.dumps(meas.get("delta_counts", {})),
        left_copied=int(meta["copy_left"].get("copied", 0)),
        right_copied=int(meta["copy_right"].get("copied", 0)),
        right_skipped_overlap=int(meta["copy_right"].get("skipped_overlap", 0)),
        right_incompatible=int(meta["copy_right"].get("incompatible", 0)),
    )
    for name, rr in regions.items():
        row.update(_region_metrics(name, rr))
    row.update(_seam_mismatch(meas, orientation, offset))
    row.update(_wilson_crossing_summary(meas, orientation, offset, max_loop_side=max_loop_side))
    row["clean_fusion"] = bool(row["z2_fraction"] >= 0.999)
    row["localized_seam_defect"] = bool(row["interior_z2_fraction"] >= 0.90 and row["seam_z2_fraction"] < row["interior_z2_fraction"] - 0.15)
    row["seam_residue"] = float(max(0.0, row["interior_z2_fraction"] - row["seam_z2_fraction"]))
    row["fusion_score"] = float(1.0*row["z2_fraction"] + 0.75*row["interior_z2_fraction"] + 0.50*row["seam_z2_fraction"] - 0.25*row["seam_residue"])
    return row, fused


# --------------------------------------------------------------------------- #
# Optional seam repair selection
# --------------------------------------------------------------------------- #
def _seam_vertices(lat: ML.MicroLattice, orientation: Orientation, offset: int, band: int = 1) -> set[int]:
    out: set[int] = set()
    if orientation == "horizontal":
        cols = set(range(max(0, int(offset)-int(band)), min(int(lat.nx), int(offset)+int(band)) + 1))
        for (r,c), vs in lat.node_out.items():
            if int(c) in cols:
                out.update(int(v) for v in vs if len(lat.joint.preds[int(v)]) > 0)
        for key, ed in lat.edges.items():
            src, dst = key
            if int(src[1]) in cols or int(dst[1]) in cols:
                for attr in ("branch_left", "branch_right", "panel_left", "panel_right"):
                    out.update(int(v) for v in getattr(ed, attr) if len(lat.joint.preds[int(v)]) > 0)
    else:
        rows = set(range(max(0, int(offset)-int(band)), min(int(lat.ny), int(offset)+int(band)) + 1))
        for (r,c), vs in lat.node_out.items():
            if int(r) in rows:
                out.update(int(v) for v in vs if len(lat.joint.preds[int(v)]) > 0)
        for key, ed in lat.edges.items():
            src, dst = key
            if int(src[0]) in rows or int(dst[0]) in rows:
                for attr in ("branch_left", "branch_right", "panel_left", "panel_right"):
                    out.update(int(v) for v in getattr(ed, attr) if len(lat.joint.preds[int(v)]) > 0)
    return out


def _clone_lat(lat: ML.MicroLattice) -> ML.MicroLattice:
    return BM.clone_lattice(lat)


def _mutate_vertices(lat: ML.MicroLattice, vertices: Iterable[int], rng: np.random.Generator, entry_rate: float, table_rate: float) -> ML.MicroLattice:
    out = _clone_lat(lat)
    vs = list(int(v) for v in vertices if len(out.joint.preds[int(v)]) > 0)
    touched = 0
    for v in vs:
        if rng.random() < float(entry_rate):
            out.joint.rules[v] = _mutate_rule(out.joint.rules[v], int(out.q), float(table_rate), rng)
            touched += 1
    if vs and touched == 0:
        v = int(rng.choice(vs))
        out.joint.rules[v] = _mutate_rule(out.joint.rules[v], int(out.q), max(float(table_rate), 1.0/max(1, out.joint.rules[v].size)), rng)
    return out


def repair_fusion(
    fused: ML.MicroLattice,
    orientation: Orientation,
    offset: int,
    rng: np.random.Generator,
    generations: int = 0,
    population: int = 16,
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    entry_rate: float = 0.06,
    table_rate: float = 0.08,
    seam_band: int = 1,
) -> tuple[dict, ML.MicroLattice]:
    if generations <= 0:
        meas = ML.measure_microscopic_lattice(fused, initial_mode=initial_mode, n_random_initial=n_random_initial, rng=rng)
        regions = _region_rows(meas, orientation, offset)
        row = dict(z2_fraction=float(meas["z2_fraction"]), seam_z2_fraction=_region_metrics("seam", regions["seam"])["seam_z2_fraction"], interior_z2_fraction=_region_metrics("interior", regions["interior"])["interior_z2_fraction"])
        row["repair_score"] = row["z2_fraction"] + row["seam_z2_fraction"]
        return row, fused
    seam_vs = _seam_vertices(fused, orientation, offset, band=seam_band)
    pop = [_clone_lat(fused)]
    for _ in range(max(0, int(population)-1)):
        pop.append(_mutate_vertices(fused, seam_vs, rng, entry_rate, table_rate))
    best_lat = pop[0]; best_row: dict | None = None; best_score = -1e9
    for gen in range(int(generations)+1):
        scored = []
        for cand in pop:
            meas = ML.measure_microscopic_lattice(cand, initial_mode=initial_mode, n_random_initial=n_random_initial, rng=rng)
            regions = _region_rows(meas, orientation, offset)
            row = dict(z2_fraction=float(meas["z2_fraction"]), valid_fraction=float(meas["valid_fraction"]), nontrivial_fraction=float(meas["nontrivial_fraction"]))
            for name, rr in regions.items():
                row.update(_region_metrics(name, rr))
            row.update(_seam_mismatch(meas, orientation, offset))
            score = float(1.0*row["z2_fraction"] + 0.75*row["interior_z2_fraction"] + 0.75*row["seam_z2_fraction"] - 0.25*max(0.0, row["interior_z2_fraction"]-row["seam_z2_fraction"]))
            row["repair_score"] = score; row["repair_generation"] = int(gen)
            scored.append((score, cand, row))
            if score > best_score:
                best_score = score; best_lat = cand; best_row = dict(row)
        if gen == int(generations):
            break
        scored.sort(key=lambda t: t[0], reverse=True)
        elites = [t[1] for t in scored[:max(1, int(population)//4)]]
        new_pop = [_clone_lat(elites[0])]
        while len(new_pop) < int(population):
            parent = elites[int(rng.integers(0, len(elites)))]
            new_pop.append(_mutate_vertices(parent, seam_vs, rng, entry_rate, table_rate))
        pop = new_pop
    return best_row or {}, best_lat


# --------------------------------------------------------------------------- #
# Sweep and summary
# --------------------------------------------------------------------------- #
def _pair_indices(n: int, pair_samples: int, rng: np.random.Generator, mode: str = "random") -> list[tuple[int,int]]:
    n = int(n)
    if n < 2:
        return []
    if mode == "all":
        pairs = [(i,j) for i in range(n) for j in range(n) if i != j]
        return pairs[:int(pair_samples)] if int(pair_samples) > 0 else pairs
    out = []
    for _ in range(int(pair_samples)):
        i = int(rng.integers(0, n)); j = int(rng.integers(0, n-1))
        if j >= i: j += 1
        out.append((i,j))
    return out


def run_lattice_fusion_audit(
    winner_path: str,
    top_n: int = 20,
    pair_samples: int = 80,
    pair_mode: str = "random",
    orientation: Orientation = "horizontal",
    initial_mode: InitialMode = "joint_random",
    n_random_initial: int = 4096,
    max_loop_side: int = 4,
    background_ensemble: str = "random",
    repair_generations: int = 0,
    population: int = 16,
    entry_rate: float = 0.06,
    table_rate: float = 0.08,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    winners = load_lattice_winners(winner_path, top_n=top_n)
    if len(winners) < 2:
        raise ValueError("need at least two lattice winners to fuse")
    rng = np.random.default_rng(int(base_seed))
    pairs = _pair_indices(len(winners), pair_samples, rng, mode=pair_mode)
    rows: list[dict] = []
    saved: list[dict] = []
    for n, (i, j) in enumerate(pairs):
        seed = (int(base_seed)*1000003 + i*9176 + j*8191 + n) % 2**32
        rrng = np.random.default_rng(seed)
        left = winners[i]["lattice"]; right = winners[j]["lattice"]
        row, fused = evaluate_fusion(left, right, orientation=orientation, rng=rrng, initial_mode=initial_mode, n_random_initial=n_random_initial, max_loop_side=max_loop_side, background_ensemble=background_ensemble)
        row.update(pair_index=int(n), left_index=int(i), right_index=int(j), winner_path=str(winner_path), repair_generations=int(repair_generations))
        if int(repair_generations) > 0:
            rep, repaired = repair_fusion(fused, orientation, int(row["offset"]), rrng, generations=int(repair_generations), population=int(population), initial_mode=initial_mode, n_random_initial=n_random_initial, entry_rate=entry_rate, table_rate=table_rate)
            for k, v in rep.items():
                row[f"repair_{k}"] = v
            row["repair_clean_fusion"] = bool(row.get("repair_z2_fraction", 0.0) >= 0.999)
            saved_lat = repaired
        else:
            saved_lat = fused
        rows.append(row)
        saved.append(dict(metrics=dict(row), lattice=saved_lat))
        if verbose and (n % max(1, len(pairs)//10 or 1) == 0):
            print(f"lattice-fusion pair {n+1}/{len(pairs)} i={i} j={j} done", flush=True)
    return pd.DataFrame(rows), saved


def _merged_json_counts(series: pd.Series) -> dict:
    out: dict[str, int] = {}
    for s in series.dropna().tolist():
        try:
            d = json.loads(s) if isinstance(s, str) else dict(s)
        except Exception:
            d = {}
        for k, v in d.items():
            out[str(k)] = out.get(str(k), 0) + int(v)
    return out


def summarize_fusion(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(verdict="NO DATA", n_rows=0)
    clean = float(df.get("clean_fusion", pd.Series(dtype=bool)).mean()) if "clean_fusion" in df else 0.0
    local = float(df.get("localized_seam_defect", pd.Series(dtype=bool)).mean()) if "localized_seam_defect" in df else 0.0
    seam_res = float(df["seam_residue"].mean()) if "seam_residue" in df else math.nan
    seam_z2 = float(df["seam_z2_fraction"].mean()) if "seam_z2_fraction" in df else math.nan
    interior_z2 = float(df["interior_z2_fraction"].mean()) if "interior_z2_fraction" in df else math.nan
    all_z2 = float(df["z2_fraction"].mean()) if "z2_fraction" in df else math.nan
    if clean > 0.5:
        verdict = "CLEAN VACUUM FUSION: independently selected Z2 lattices usually fuse without seam residue"
    elif local > 0.25 or (math.isfinite(seam_res) and seam_res > 0.15 and interior_z2 > 0.8):
        verdict = "INTERFACE DEFECT SIGNAL: fused selected vacua leave localized seam consistency residues"
    elif all_z2 > 0.7:
        verdict = "PARTIAL VACUUM FUSION: fused vacua mostly retain Z2 structure but defects are not sharply localized"
    else:
        verdict = "NO ROBUST VACUUM FUSION: selected vacua do not coherently fuse in this audit"
    summary = dict(
        verdict=verdict,
        n_rows=int(len(df)),
        clean_fusion_fraction=clean,
        localized_seam_defect_fraction=local,
        mean_z2_fraction=all_z2,
        mean_left_z2_fraction=float(df["left_z2_fraction"].mean()) if "left_z2_fraction" in df else math.nan,
        mean_right_z2_fraction=float(df["right_z2_fraction"].mean()) if "right_z2_fraction" in df else math.nan,
        mean_interior_z2_fraction=interior_z2,
        mean_seam_z2_fraction=seam_z2,
        mean_seam_residue=seam_res,
        mean_seam_pair_z2_fraction=float(df["seam_pair_z2_fraction"].mean()) if "seam_pair_z2_fraction" in df else math.nan,
        mean_seam_flux_mismatch_fraction=float(df["seam_flux_mismatch_fraction"].dropna().mean()) if "seam_flux_mismatch_fraction" in df and len(df["seam_flux_mismatch_fraction"].dropna()) else math.nan,
        mean_wilson_crossing_abs=float(df["wilson_crossing_abs"].dropna().mean()) if "wilson_crossing_abs" in df and len(df["wilson_crossing_abs"].dropna()) else math.nan,
        mean_wilson_noncrossing_abs=float(df["wilson_noncrossing_abs"].dropna().mean()) if "wilson_noncrossing_abs" in df and len(df["wilson_noncrossing_abs"].dropna()) else math.nan,
        group_name_counts=_merged_json_counts(df.get("group_name_counts", pd.Series(dtype=str))),
        delta_type_counts=_merged_json_counts(df.get("delta_type_counts", pd.Series(dtype=str))),
    )
    if "repair_z2_fraction" in df:
        summary.update(
            repair_mean_z2_fraction=float(df["repair_z2_fraction"].mean()),
            repair_clean_fusion_fraction=float(df.get("repair_clean_fusion", pd.Series(dtype=bool)).mean()),
            repair_mean_seam_z2_fraction=float(df["repair_seam_z2_fraction"].mean()) if "repair_seam_z2_fraction" in df else math.nan,
        )
    return summary


def plot_fusion(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if df.empty:
        fig.savefig(path); plt.close(fig); return
    ax[0].hist(df["seam_residue"].dropna(), bins=20)
    ax[0].set_title("seam residue")
    ax[0].set_xlabel("interior_z2 - seam_z2")
    ax[0].set_ylabel("pairs")
    ax[1].scatter(df["interior_z2_fraction"], df["seam_z2_fraction"], c=df["z2_fraction"], s=35)
    ax[1].set_title("interior vs seam validity")
    ax[1].set_xlabel("interior Z2 fraction")
    ax[1].set_ylabel("seam Z2 fraction")
    if "repair_z2_fraction" in df:
        vals = [df["z2_fraction"].mean(), df["repair_z2_fraction"].mean()]
        labs = ["raw", "repaired"]
    else:
        vals = [df["left_z2_fraction"].mean(), df["seam_z2_fraction"].mean(), df["right_z2_fraction"].mean()]
        labs = ["left", "seam", "right"]
    ax[2].bar(labs, vals)
    ax[2].set_title("fusion validity summary")
    ax[2].set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_fusion_winners(path: str, records: list[dict], top_n: int = 50) -> None:
    ranked = sorted(records, key=lambda r: (float(r.get("metrics", {}).get("repair_z2_fraction", r.get("metrics", {}).get("z2_fraction", 0.0))), float(r.get("metrics", {}).get("seam_z2_fraction", 0.0)), -float(r.get("metrics", {}).get("seam_residue", 0.0))), reverse=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(kind="lattice_fusion_winners", winners=ranked[:int(top_n)]), f)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fuse independently selected Z2 lattice vacua and audit seam defects.")
    p.add_argument("winner_path")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--pair-samples", type=int, default=80)
    p.add_argument("--pair-mode", choices=["random", "all"], default="random")
    p.add_argument("--orientation", choices=["horizontal", "vertical"], default="horizontal")
    p.add_argument("--initial-mode", choices=["source_all", "joint_random"], default="joint_random")
    p.add_argument("--n-random-initial", type=int, default=4096)
    p.add_argument("--max-loop-side", type=int, default=4)
    p.add_argument("--background-ensemble", default="random")
    p.add_argument("--repair-generations", type=int, default=0)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--entry-rate", type=float, default=0.06)
    p.add_argument("--table-rate", type=float, default=0.08)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--save-winners", default=None)
    p.add_argument("--save-winner-top-n", type=int, default=50)
    p.add_argument("--out", default="example_results/lattice_fusion.csv")
    p.add_argument("--plot", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    df, records = run_lattice_fusion_audit(
        args.winner_path, top_n=int(args.top_n), pair_samples=int(args.pair_samples), pair_mode=args.pair_mode,
        orientation=args.orientation, initial_mode=args.initial_mode, n_random_initial=int(args.n_random_initial),
        max_loop_side=int(args.max_loop_side), background_ensemble=args.background_ensemble,
        repair_generations=int(args.repair_generations), population=int(args.population),
        entry_rate=float(args.entry_rate), table_rate=float(args.table_rate), base_seed=int(args.base_seed), verbose=not bool(args.quiet),
    )
    summary = summarize_fusion(df)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        plot_fusion(df, args.plot)
    if args.save_winners:
        save_fusion_winners(args.save_winners, records, top_n=int(args.save_winner_top_n))
    print(json.dumps(summary, indent=2))
    if args.out:
        print(f"wrote {args.out}")
        print(f"wrote {os.path.splitext(args.out)[0]}_summary.json")
    if args.plot:
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
