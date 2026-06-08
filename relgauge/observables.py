"""
observables.py -- instance-level measurements and the structural-theorem checks.

For ONE relational system we report a dictionary of measures:
  partition survival (full / image-restricted, count and log),
  orbit dimension (image / recurrent),
  resolvability R (active bisimulation; passive horizon-h), recurrent-restricted,
  plus the recurrent-set size and the 1/e reference.

Separately we provide falsifiable structural checks:
  S1 condensation-is-DAG, S2 finite light-cone, S3 fusion monotonicity.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import core as C
from . import quotients as Q


# --------------------------------------------------------------------------- #
# all per-instance measures
# --------------------------------------------------------------------------- #
def measure_instance(sys: C.RelationalSystem, w: int,
                     passive_horizons=(1, 2),
                     do_active=True, do_partition=True) -> dict:
    k, q = sys.k, sys.q
    sm = sys.all_step_maps()
    adj = C.orbit_adjacency_fast(sm)
    rec = C.recurrent_states(adj)
    img = C.image_states(sm)
    out = dict(k=k, q=q, w=w, ratio=w / k,
               n_rec=int(len(rec)), n_img=int(len(img)),
               qk=q ** k, one_over_e_ref=(1 - 1 / q) ** q)

    boundary = tuple(range(min(w, k)))

    # ---- partition survival ----
    if do_partition:
        labels = Q.partition_quotient(sys, sm)
        full = []
        imgc = []
        for v in range(k):
            imgvals = set(((img // (q ** v)) % q).tolist())
            full.append(Q.partition_class_count(labels, v) / q)
            denom = max(1, len(imgvals))
            imgc.append(Q.partition_class_count(labels, v, imgvals) / denom)
        out["part_surv_full_count"] = float(np.mean(full))
        out["part_surv_image_count"] = float(np.mean(imgc))

    # ---- orbit dimension ----
    out["orbit_dim_image"] = float(Q.orbit_dimension(sys, sm, "image"))
    out["orbit_dim_recurrent"] = float(Q.orbit_dimension(sys, sm, "recurrent"))

    # ---- resolvability (needs >=2 recurrent states to normalize) ----
    if len(rec) >= 2:
        logrec = np.log(len(rec))
        if do_active:
            block = Q.boundary_bisimulation(sys, boundary, sm)
            ncl = len(set(block[rec].tolist()))
            out["R_active"] = float(np.log(ncl) / logrec)
            out["R_active_classes"] = int(ncl)
        for h in passive_horizons:
            cls = Q.passive_signature_classes(sys, boundary, h, sm)
            ncl = len(set(cls[rec].tolist()))
            out[f"R_passive_h{h}"] = float(np.log(ncl) / logrec)
    else:
        out["R_active"] = np.nan
        for h in passive_horizons:
            out[f"R_passive_h{h}"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# S2  finite light cone
# --------------------------------------------------------------------------- #
def check_light_cone(sys: C.RelationalSystem, trials: int,
                     rng: np.random.Generator, mode: str = "sync") -> bool:
    """Does a single-vertex flip stay within one edge after one tick?

    mode='sync'  : synchronous update (all read old values).  The structural
                   'speed = 1 edge/tick' claim should HOLD here.
    mode='async' : a fixed sequential schedule.  The claim generally FAILS,
                   because a tick can propagate along a path -- this is the
                   honest status of the finite-light-cone claim under the
                   schedule (gauge) semantics the rest of the theory uses.
    """
    k, q = sys.k, sys.q
    out_nb = {u: set() for u in range(k)}
    for v in range(k):
        for p in sys.preds[v]:
            out_nb[p].add(v)
    if mode == "sync":
        sm = C.synchronous_step_map(sys)
    else:
        sm = sys.step_map(tuple(range(k)))
    S = C.all_states(k, q)
    for _ in range(trials):
        x = int(rng.integers(0, q ** k))
        u = int(rng.integers(0, k))
        st2 = S[x].copy()
        st2[u] = (st2[u] + int(rng.integers(1, q))) % q
        a = S[sm[x]]
        b = S[sm[C.encode(st2[None, :], q)[0]]]
        diff = np.where(a != b)[0]
        allowed = out_nb[u] | ({u} if u in sys.preds[u] else set())
        for d in diff:
            if int(d) not in allowed:
                return False
    return True


# --------------------------------------------------------------------------- #
# S3  fusion monotonicity of hidden-multiplicity entropy
# --------------------------------------------------------------------------- #
def _hidden_multiplicities(F, X_visible_idx, hidden_dims, q, k, O, H,
                           combine):
    """Return dict (xO, xpO) -> count of hidden assignments realising the
    visible transition xO -> xpO under deterministic map F on X_P."""
    # enumerate visible states and hidden states
    counts = {}
    nO, nH = len(O), len(H)
    Xvis = C.all_states(nO, q) if nO else np.zeros((1, 0), int)
    Xhid = C.all_states(nH, q) if nH else np.zeros((1, 0), int)
    for vo in range(Xvis.shape[0]):
        xO = tuple(Xvis[vo].tolist())
        for vh in range(Xhid.shape[0]):
            full = combine(xO, tuple(Xhid[vh].tolist()))
            outfull = F(full)
            xpO = tuple(outfull[o] for o in O)
            counts[(xO, xpO)] = counts.get((xO, xpO), 0) + 1
    return counts


def fusion_monotonicity_event(kP, q, r, rng, feedback=False):
    """Construct P (SCC) and P' = P + r fused vertices, compare hidden
    multiplicity entropies for every visible transition.

    feedback=False : fused vertices read only from P (valid extension);
                     theorem predicts |M'| = |M|*q^r  (>= holds, exactly).
    feedback=True  : fused vertices ALSO feed back into P (violates the
                     extension hypothesis); >= may FAIL -- used to show the
                     hypothesis is necessary.

    Returns (min_ratio, all_ge) where ratio = |M'| / |M| over shared
    visible transitions, and all_ge is whether |M'| >= |M| everywhere.
    """
    from . import core as C
    # P : a cycle of size kP with random rules (single SCC)
    P = C.make_cycle(kP, q, rng)
    O = tuple(range(max(1, kP // 2)))      # visible slice
    H = tuple(v for v in range(kP) if v not in O)

    schedP = tuple(range(kP))
    smP = P.step_map(schedP)
    SP = C.all_states(kP, q)

    def FP(full):  # full is length-kP tuple
        idx = C.encode(np.array(full, dtype=np.int64)[None, :], q)[0]
        return SP[smP[idx]].tolist()

    def combineP(xO, xH):
        full = [0] * kP
        for j, o in enumerate(O):
            full[o] = xO[j]
        for j, h in enumerate(H):
            full[h] = xH[j]
        return tuple(full)

    countsP = _hidden_multiplicities(FP, None, None, q, kP, O, H, combineP)

    # P' : add r downstream vertices reading from P (and optionally feeding back)
    kPp = kP + r
    preds = [list(P.preds[v]) for v in range(kP)] + [[] for _ in range(r)]
    for j in range(r):
        nv = kP + j
        src = int(rng.integers(0, kP))
        preds[nv].append(src)
        if feedback:
            # new vertex also feeds into a P-vertex -> alters P's update
            tgt = int(rng.integers(0, kP))
            preds[tgt].append(nv)
    preds = [tuple(sorted(set(p))) for p in preds]
    rules = []
    for v in range(kPp):
        d = len(preds[v])
        if v < kP and not feedback:
            # keep P's original rule but it may have new (no) predecessors; in
            # the no-feedback case P's preds are unchanged so reuse rule
            rules.append(P.rules[v])
        else:
            rules.append(rng.integers(0, q, size=q ** d, dtype=np.int64))
    Pp = C.RelationalSystem(kPp, q, preds, rules)

    schedPp = tuple(range(kPp))
    smPp = Pp.step_map(schedPp)
    SPp = C.all_states(kPp, q)
    Hp = tuple(v for v in range(kPp) if v not in O)

    def FPp(full):
        idx = C.encode(np.array(full, dtype=np.int64)[None, :], q)[0]
        return SPp[smPp[idx]].tolist()

    def combinePp(xO, xH):
        full = [0] * kPp
        for j, o in enumerate(O):
            full[o] = xO[j]
        for j, h in enumerate(Hp):
            full[h] = xH[j]
        return tuple(full)

    countsPp = _hidden_multiplicities(FPp, None, None, q, kPp, O, Hp, combinePp)

    ratios = []
    all_ge = True
    for key, mP in countsP.items():
        mPp = countsPp.get(key, 0)
        if mPp < mP:
            all_ge = False
        if mP > 0:
            ratios.append(mPp / mP)
    return (min(ratios) if ratios else float("nan")), all_ge
