# Resolvability experiment -- falsification report

_generated 2026-06-07 13:15:50; python 3.12.3, numpy 2.4.4_

_instances total: 1440_


## 1. Validation of established results (code self-checks)

**Partition collapse on cycles** (image/recurrent-restricted survival should fall toward 0 as q grows; full-alphabet should track $1/q+(1-1/q)^q$):\n

| q | image-restricted | full-alphabet | $1/q+(1-1/q)^q$ |
|---|---|---|---|
| 2 | 0.746 | 0.746 | 0.750 |
| 4 | 0.396 | 0.577 | 0.566 |
| 6 | 0.260 | 0.499 | 0.502 |
| 8 | 0.196 | 0.466 | 0.469 |

**Orbit dimension on cycles** (should approach (k-1)/k):


| k | measured | (k-1)/k |
|---|---|---|
| 2 | 0.513 | 0.500 |
| 3 | 0.623 | 0.667 |
| 4 | 0.648 | 0.750 |

**Structural checks:**

- S1 condensation is a DAG (acyclic causality between observers): PASS (60 systems).
- S2 finite light-cone: holds SYNCHRONOUSLY 100%, holds ASYNCHRONOUSLY 10%.  => the 'speed = 1 edge/tick' claim is a synchronous property and FAILS under the schedule (gauge) semantics.
- S3 fusion monotonicity: |M'|>=|M| in 40/40 valid-extension events (min ratio ~ q^r); violated in 13/40 events when the extension hypothesis is broken (so the hypothesis is necessary, as proved).

## 2. The open question: resolvability scaling

Claim under test: with rho = w/k, the limit R* = lim_{rho->0} mean R exists, is > 0, and is q- and ensemble-independent.


> **Cardinality caveat (read first).** A *fixed* finite resolver must resolve a vanishing *fraction* of a growing system (pigeonhole); so R trending to 0 as rho->0 is the expected null, not a surprise. The non-trivial possibilities are: (i) a nonzero limit (genuine relational invariant), or (ii) a universal *slope* R/rho (intensive invariant). Both are reported below.


### R_active

- **cycle** (largest k=4, i.e. smallest rho=1/4):
    - R at smallest rho, by q: q2=1.000, q4=1.000, q6=1.000, q8=1.000
    - q-independence: spread=0.000 (FLAT), corr with (1-1/q)^q = nan
    - trend in k (at largest q): non-monotone, first=1.000 -> last=1.000, last step delta=0.000
- **scc** (largest k=4, i.e. smallest rho=1/4):
    - R at smallest rho, by q: q2=0.670, q4=0.909, q6=0.943, q8=0.958
    - q-independence: spread=0.288 (DRIFTS), corr with (1-1/q)^q = 0.99
    - trend in k (at largest q): decreasing, first=1.000 -> last=0.958, last step delta=-0.020

### R_passive_h1

- **cycle** (largest k=4, i.e. smallest rho=1/4):
    - R at smallest rho, by q: q2=0.508, q4=0.513, q6=0.526, q8=0.550
    - q-independence: spread=0.041 (DRIFTS), corr with (1-1/q)^q = 0.76
    - trend in k (at largest q): decreasing, first=1.000 -> last=0.550, last step delta=-0.211
- **scc** (largest k=4, i.e. smallest rho=1/4):
    - R at smallest rho, by q: q2=0.433, q4=0.628, q6=0.695, q8=0.745
    - q-independence: spread=0.312 (DRIFTS), corr with (1-1/q)^q = 1.00
    - trend in k (at largest q): decreasing, first=1.000 -> last=0.745, last step delta=-0.157

### R_passive_h2

- **cycle** (largest k=4, i.e. smallest rho=1/4):
    - R at smallest rho, by q: q2=0.657, q4=0.664, q6=0.685, q8=0.723
    - q-independence: spread=0.066 (DRIFTS), corr with (1-1/q)^q = 0.75
    - trend in k (at largest q): decreasing, first=1.000 -> last=0.723, last step delta=-0.220
- **scc** (largest k=4, i.e. smallest rho=1/4):
    - R at smallest rho, by q: q2=0.509, q4=0.753, q6=0.824, q8=0.885
    - q-independence: spread=0.376 (DRIFTS), corr with (1-1/q)^q = 1.00
    - trend in k (at largest q): decreasing, first=1.000 -> last=0.885, last step delta=-0.074

## 3. Automated verdict

- **R_active: ARTIFACT (q-dependent / 1-over-e correlated)**
- **R_passive_h1: ARTIFACT (q-dependent / 1-over-e correlated)**
- **R_passive_h2: ARTIFACT (q-dependent / 1-over-e correlated)**

_Interpretation key_: SUPPORTED = nonzero, q-flat, not artifact-correlated, converging; TRIVIAL = R->0 as rho->0; ARTIFACT = nonzero but q-drifting / correlated with (1-1/q)^q; INCONCLUSIVE = trend present but not converged within reach (larger k needed -- exponentially expensive).


Figures: `fig_partition_collapse.png`, `fig_orbit_dim.png`, `fig_scaling_*.png`.
