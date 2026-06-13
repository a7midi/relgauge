# Observer section dynamics patch

This patch adds the first concrete multi-cycle arena for observer-frame section-obstruction dynamics.

## New graph modes

`observerframebundle.py`, `blindobserverconnection.py`, and `observerconnectionsuccessaudit.py` now understand these controlled observer-relative geometries:

```text
frame_theta_flat_c2
frame_theta_twist_c2
frame_random_theta
frame_double_flat_c2
frame_double_twist_c2
frame_random_double_diamond
```

The theta graph has three observer paths from one source observer to one sink observer. It has two independent loops. With one branch twist, two pairwise loops carry nontrivial C2 holonomy, as expected.

The double-diamond graph has two coupled diamonds in series. It is a small arena where section-obstruction representatives can potentially shift between two loops.

## New blind-selection option

`blindobserverconnection.py` has a new flag:

```bash
--cycle-objective best|all
```

`best` preserves the single-diamond v4 objective. `all` is intended for theta and double-diamond runs: it rewards the fraction of discovered loops/chords that close as automorphisms. It still does not reward C2, nontrivial holonomy, group order, flux, particles, or motion.

## New module

```text
relgauge/observersectiondynamics.py
```

This module audits section-gluing residuals over a selected/certified observer-frame connection.

For each microscopic state `x` and each true live frame transport edge `e:O->P`, it computes:

```text
r_e(x) = 0 if target frame value equals U_e(source frame value)
       = 1 otherwise
```

It then evolves those residual patterns under sampled admissible update schedules and computes the deterministic quotient forced by schedule ambiguity.

The module is diagnostic only. It does not reward motion, lifetime, flux, particles, or nontriviality.

## Example controls

Theta nonflat control:

```bash
python -m relgauge.observerframebundle 4 ^
  --graphs 1 ^
  --graph-mode frame_theta_twist_c2 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/observer_frame_theta_twist_c2.csv
```

Double-diamond nonflat control:

```bash
python -m relgauge.observerframebundle 4 ^
  --graphs 1 ^
  --graph-mode frame_double_twist_c2 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/observer_frame_double_twist_c2.csv
```

Seeded theta recovery using the all-loop objective:

```bash
python -m relgauge.blindobserverconnection 4 ^
  --graph-mode frame_random_theta ^
  --seed-mode theta_twist_c2 ^
  --cycle-objective all ^
  --population 8 ^
  --generations 4 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/seeded_theta_connection.csv ^
  --winners example_results/seeded_theta_connection_winners.pkl
```

Section dynamics on the certified single diamond:

```bash
python -m relgauge.observersectiondynamics 4 ^
  --winners example_results/blind_observer_connection_v4_q4_winners.pkl ^
  --winner-index 0 ^
  --frame-coordinate-mode local_charts ^
  --max-state-samples 65536 ^
  --schedule-samples 16 ^
  --out example_results/section_dynamics_certified_diamond.csv
```

Expected interpretation: static one-loop frustration, not motion.

Section dynamics on a theta control:

```bash
python -m relgauge.observersectiondynamics 4 ^
  --graph-mode frame_theta_twist_c2 ^
  --frame-coordinate-mode local_charts ^
  --max-state-samples 4096 ^
  --schedule-samples 16 ^
  --out example_results/section_dynamics_theta_twist.csv
```

This is the first meaningful arena for possible motion. The diagnostic will report whether residual representatives have a nontrivial recurrent schedule-invariant quotient.
