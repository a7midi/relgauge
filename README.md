# relgauge — schedule-gauge resolvability in finite relational systems

A self-contained, reproducible experimental package to **validate or falsify**
the reformulated theory we converged on: *in a closed finite relational system,
the physics is the invariant structure that a finite interior part can resolve
about the rest.* It is built to be run locally and to produce
publication-grade, seed-reproducible results.

---

## 1. The object under study

A **finite relational dynamical system** is a directed multigraph `G=(V,E)`
with a finite local state space `[q]` at each vertex and a deterministic local
rule `λ_v : [q]^{In(v)} → [q]`. For the resolvability experiments we work with a
single **strongly connected component** `S` (one "observer"): every vertex's
predecessors lie inside `S`, so all `k!` vertex orderings are admissible
schedules. `T_σ : [q]^k → [q]^k` is the in-place sequential one-tick map under
ordering `σ`.

The **schedule is gauge** (closure principle: a closed finite system has no
external frame, and finite interior observers cannot invert it). The physical
dynamics is therefore the **nondeterministic lift** `N(x) = {T_σ(x) : σ}`.

Three quotients of the schedule freedom (all implemented):

| quotient | meaning | behaviour (proved + reproduced here) |
|---|---|---|
| **partition** `Λ*` | strict gauge: merge anything orderings disagree on | collapses cycles to a point on the recurrent set (survival → 0) — *too aggressive* |
| **orbit** (intrinsic) | track every ordering outcome | reachable dimension `→ (k-1)/k → 1` on cycles — *too generous* |
| **boundary resolvability** | what a finite resolver distinguishes through an interface | the quantity of interest (below) |

---

## 2. The resolvability quantity (precise definition)

Let `Rec(S)` be the **recurrent set** of the lift `N` (states on a directed
cycle of the reachability graph). Restricting to `Rec(S)` neutralizes the
`(1-1/q)^q` image-inflation artifact (non-image states survive trivially).

Let `∂ ⊆ V` be a **boundary** of width `w = |∂|` — what a downstream resolver
reads. Because the schedule is gauge, the resolver sees the boundary-labeled
nondeterministic LTS: states colored by `π_∂(x)`, transitions `x —π_∂(y)→ y`
for `y ∈ N(x)`. Two states are **resolver-equivalent** under the largest
bisimulation `~` of this LTS (active/interactive resolver; a passive
finite-horizon variant is also computed as a conservative bracket).

> **Resolvability**
> ```
>            log #{ [x]~ : x ∈ Rec(S) }
>   R  =  ───────────────────────────────   ∈ [0,1]
>                log #Rec(S)
> ```
> the fraction of genuinely-recurrent state distinctions that survive to the
> boundary.

**Size ratio** `ρ = w/k`. The scaling object is `R̄(k,q,w,ensemble) = E[R]`,
studied as `ρ → 0` (grow `k`, hold `w`).

### The single falsifiable claim (no constant baked in)
> `R* = lim_{ρ→0} R̄(k,q,w,ensemble)` **exists, is > 0, and is independent of
> `q` and of the ensemble.**

**Outcome grid** (decided automatically by `report.py`):
- **SUPPORTED** — nonzero, `q`-flat, not artifact-correlated, converged.
- **TRIVIAL** — `R → 0` as `ρ → 0` (observers resolve nothing in the limit).
- **ARTIFACT** — nonzero but drifts with `q` / correlates with `(1-1/q)^q`.
- **INCONCLUSIVE** — trend present but not converged within computational reach.

> **Cardinality caveat (built into the report).** A *fixed* finite resolver
> must resolve a vanishing *fraction* of a growing system (pigeonhole), so
> `R → 0` is the expected null, not a discovery. The two non-trivial
> possibilities the report tracks are a **nonzero limit** (genuine relational
> invariant) and a **universal slope** `R/ρ` (intensive invariant). If neither
> appears, the strong reading of the theory is falsified and only the weak
> reading (physics = observer-relative, scale-dependent resolvable structure,
> with no universal number) survives.

---

## 3. What the package already establishes (self-validation)

`tests/test_theory.py` asserts the results we proved analytically; they double
as the report's "Section 1" self-checks and pass on every run:

- **Partition collapse** — image/recurrent-restricted survival on cycles `→ 0`
  (`~ 1/((1-1/e)q)`); full-alphabet survival `= 1/q + (1-1/q)^q` (the artifact).
- **Orbit dimension** — reachable dimension on cycles `→ (k-1)/k`.
- **S1** — the SCC condensation is acyclic (acyclic causality between observers).
- **S2** — finite light-cone holds **synchronously** but **fails asynchronously**
  (it is a synchronous property; the "speed = 1 edge/tick" claim does not
  survive the schedule/gauge semantics — a genuine correction the tool surfaces).
- **S3** — hidden-multiplicity entropy is **monotone under valid fusion**
  (ratio `= q^r`), and the extension hypothesis is **necessary** (monotonicity
  can fail with feedback).

If any of these breaks, the code or a theorem is wrong — that is the point.

---

## 4. Install & run

Requirements: Python ≥ 3.10, `numpy`, `pandas`, `matplotlib` (`scipy`/`pytest`
optional). Everything is pure CPU.

```bash
pip install numpy pandas matplotlib
cd relgauge

# fast smoke test (~1-2 min): reproduces every theorem + a preliminary verdict
python -m relgauge.run --preset quick --out results_quick

# publication run (hours): wider k,q; multiple ensembles & boundary widths
python -m relgauge.run --preset full --out results_full

# custom grid (overrides preset fields)
python -m relgauge.run --config my_grid.json --out results_custom

# validate the theorems
python -m pytest tests -q
```

Each run writes to `<out>/`:
`instances.csv` (one row per random system), `aggregated.csv` (bootstrap CIs per
cell), `config.json` (grid + per-instance seeds + provenance, fully
reproducible), `report.md` (the written verdict), and `fig_*.png`.

A custom config is JSON with any of:
```json
{"grid": {"ensemble": ["scc"], "k": [2,3,4,5,6,7], "q": [4,8,16,32],
          "w": [1,2], "density": [2]},
 "n_instances": 600, "passive_horizons": [1,2,3], "n_boot": 2000}
```

---

## 5. Limitations (read before publishing)

1. **Exponential ceiling.** Quotients enumerate `q^k` states and `k!` schedules.
   Practical reach is roughly `k ≤ 6–7` with moderate `q`. The `ρ → 0` limit is
   therefore only seen as a **trend over `k = 2..7`**, never directly. This is
   intrinsic to exact finite enumeration; an honest report says "inconclusive"
   rather than extrapolating. (Monte-Carlo over states, rather than full
   enumeration, is the natural extension for larger `k`.)
2. **Resolver-model dependence.** The *active* (bisimulation) resolver tends to
   saturate (`R ≈ 1`); the *passive* finite-horizon resolver gives a smaller,
   horizon-dependent fraction. There is no single canonical resolver; the
   package reports several and the scientific claim must be stated *relative to
   a declared resolver model*. This is a real conceptual open point, not a bug.
3. **Ensemble dependence is part of the test.** Universality across `cycle`,
   `scc` (tunable feedback density), and `regular` ensembles is exactly what the
   claim asserts; do not collapse them.
4. **Single-SCC scope.** The resolvability experiments treat one observer.
   Coupled-observer (resolver-as-its-own-SCC) experiments — the most faithful
   form of "made of the same stuff" — are a documented extension point
   (`observables.fusion_monotonicity_event` already builds coupled systems).

---

## 6. Module map

```
relgauge/
  core.py         systems, ensembles, SCC/condensation, sequential & sync
                  dynamics, recurrent/image sets
  quotients.py    partition Λ* (fixed point), orbit dimension,
                  boundary bisimulation, passive signatures
  observables.py  per-instance measures; S2 light-cone; S3 fusion monotonicity
  stats.py        bootstrap CIs, trend & q-independence diagnostics
  experiments.py  reproducible ensemble sweeps + aggregation
  report.py       plots + automated falsification verdict
  run.py          CLI driver (presets: quick / full)
tests/test_theory.py   the analytic results as executable assertions
```

Everything is exact finite computation; no randomness enters except the
seed-logged choice of random systems.

---

## 7. Extensions: area-law and gauge-dimension (cycle-rank) tests

Two further experiments, runnable standalone:

```bash
python -m relgauge.arealaw 4     # area-law probe (q=4): active resolver, dense SCCs
python -m relgauge.cyclerank     # gauge-deficit: single-SCC + multi-SCC additivity
```

### 7a. Area law — *falsified* in the accessible regime
We test whether the **absolute** resolvable information `I(w,k) = log2 #{[x]~}`
on `Rec` obeys an **area law** (saturates in bulk `k`, scales with boundary `w`)
or a **volume law** (tracks `log2|Rec|`; boundary transparent). Using the
*active* (maximal/infinite-horizon) resolver as the discriminator:

- **Cycles:** opacity `I/log|Rec| = 1.000` at all `k` — fully transparent.
- **Dense random SCCs:** `I_active(w=1)` *grows with* `k` and tracks `log2|Rec|`
  (opacity ≈ 0.95, only weakly decreasing) — **volume law**.

**Conclusion:** no genuine information-hiding/holographic area law for these
finite relational SCCs; the bulk state is recoverable through any boundary given
enough observation. The earlier horizon-1 "saturation" was the trivial
finite-budget effect, not hiding. **Caveat:** the edge-mode structure for the
*gauge* (the schedule is bulk-trivial and unrecoverable; cf. the partition
collapse) is real and separate — it just does **not** come with a state area
law. (The `verdict` field reports VOLUME-LAW / AREA-LAW / INCONCLUSIVE so a wider
sweep can revisit this; weak opacity drift at larger `k` is not excluded.)

### 7b. Gauge dimension — the *observer-count law* (a genuine theorem)
Define the **gauge deficit** `= |V| - log_q|reachable-in-one-step|` (the number
of `q`-ary coordinates the schedule-gauge removes, in the `q→∞` trend so the
`(1-1/e)` non-injectivity drops out of `log_q`).

- **Single SCC, varying density:** deficit ≈ **1** for `extra = 0,1,2,3`
  (β₁ = 1,2,3,4) — flat at 1 while β₁ grows. **The β₁ conjecture is refuted.**
- **Multi-SCC chains:** deficit ≈ **number of multi-vertex SCCs**
  (1, 1, 2, 2, 3 for configs `(2),(3),(2,2),(3,2),(2,2,2)`) — additive.

**Theorem (proved + reproduced):** the schedule-gauge removes **exactly one
coordinate per observer** — a single global "update phase" per feedback loop,
independent of the loop's internal density — so the total gauge dimension equals
the **number of observers** (multi-vertex SCCs), not the cycle rank. Proof: the
in-tree construction orders an SCC's vertices so every vertex but one has a
reader before it; the globally-first vertex never does (so ≥ 1), and the
construction achieves exactly 1.

This is the high-value structural result the rigor produced: it ties the gauge
content directly to the **count of observers**, which is the theory's
fundamental object — replacing the dead survival-fraction with a clean,
proved, ensemble-independent invariant.
