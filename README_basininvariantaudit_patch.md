# Basin/source invariant audit

Adds:

```text
relgauge/basininvariantaudit.py
tests/test_basininvariantaudit.py
```

This is a focused audit for an attractor-stable exact common factor. It asks whether the stable fixed-point label is correlated with the initial B boundary value.

Run:

```bash
python -m relgauge.basininvariantaudit example_results/blind_winners_q4_full.pkl ^
  --winner-index 6 ^
  --schedule-index 0 ^
  --selection-initial-pool all ^
  --selection-observer-init zero ^
  --run-initial-pool all_joint ^
  --out example_results/winner6_basin_source_charge.csv ^
  --plot example_results/fig_winner6_basin_source_charge.png
```

Main metric:

```text
mi_b_initial_boundary_to_stable_s_over_Hs
```

Interpretation:

- high: the attractor-stable label is a live basin invariant / charge candidate;
- near zero: the stable basins do not correlate with the source boundary.

The script also reports the post-B boundary version:

```text
mi_b_post_update_boundary_to_stable_s_over_Hs
```

because that value is what branches actually receive in a scheduled tick.
