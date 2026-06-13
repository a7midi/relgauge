# Observer frame bundle v4: local endpoint charts

This patch fixes the v3 representational issue where a deliberately twisted observer-frame control was flattened by shared edge common-factor coordinates.

## Key change

`observerframebundle.py` now supports:

```bash
--frame-coordinate-mode local_charts
--frame-coordinate-mode common_factor
```

`local_charts` is the default. It keeps endpoint labels local to each side of an inter-observer channel and derives the oriented edge transition map between those endpoint charts. A binary flip edge therefore remains a flip in the connection.

`common_factor` preserves the older agreement/equalizer coordinate audit. It is useful as a control because it intentionally renames both sides of an edge by the same common-factor component, which can flatten a twist.

## Positive controls

Non-flat twist control:

```bash
python -m relgauge.observerframebundle 4 \
  --graphs 1 \
  --graph-mode frame_twist_c2 \
  --vertices 8 \
  --frame-coordinate-mode local_charts \
  --out example_results/observer_frame_twist_c2_v4.csv \
  --plot example_results/fig_observer_frame_twist_c2_v4.png
```

Expected: `total_nontrivial_holonomy >= 1`, `group_counts` includes `C2`, `path_agreement = 0`.

Common-factor flattening control:

```bash
python -m relgauge.observerframebundle 4 \
  --graphs 1 \
  --graph-mode frame_twist_c2 \
  --vertices 8 \
  --frame-coordinate-mode common_factor \
  --out example_results/observer_frame_twist_c2_common_factor.csv
```

Expected: valid holonomy but trivial/identity holonomy.

Seeded twist recovery:

```bash
python -m relgauge.blindobserverconnection 4 \
  --graph-mode frame_random_diamond \
  --seed-mode twist_c2 \
  --population 12 \
  --generations 20 \
  --elite 3 \
  --mutation-rate 0.04 \
  --frame-coordinate-mode local_charts \
  --out example_results/twist_recovery_observer_connection_v4.csv \
  --winners example_results/twist_recovery_observer_connection_v4_winners.pkl
```

Expected: automorphism-valid connection with nontrivial C2 holonomy if the seed survives or is recovered.

## Tests

Focused tests:

```bash
PYTHONPATH=. pytest -q tests/test_observerframebundle.py tests/test_blindobserverconnection.py
```

Expected result from this patch: `9 passed`.
