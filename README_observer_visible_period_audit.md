# Observer visible-quotient period audit patch

This patch extends `observerhiddenmemory.py` beyond the v2 visible-quotient
schedule-absorption test.

The v2 result showed that a nontrivial quotient of visible observer states can
become schedule-deterministic while preserving the multi-cycle C2 observer-frame
connection.  However, the old summary only reported the number of recurrent
quotient classes.  One recurrent component can be either a fixed point or a
nontrivial deterministic cycle, so that field alone cannot decide whether the
schedule-consistent quotient is merely relaxing to a static attractor or has
periodic recurrent dynamics.

## New audit fields

All quotient summaries now include recurrent-period diagnostics:

```text
*_recurrent_component_count
*_recurrent_state_count
*_recurrent_component_sizes
*_cycle_lengths
*_max_period
*_fixed_point_count
*_nontrivial_periodic_recurrence
```

The fields are exposed for:

```text
memory_quotient_*
visible_schedule_quotient_*
joint_schedule_quotient_*
induced_visible_quotient_*
```

The verdict `VISIBLE-QUOTIENT PERIODIC/RECURRENT DYNAMICS CANDIDATE` is used
only when a schedule-consistent quotient has nontrivial recurrent structure or a
nontrivial period.  Motion, particles, C2, holonomy, recurrence, and localization
are still not scored by the blind selector; they are only audited post hoc.

## Recommended command

```bash
python -m relgauge.observerhiddenmemory 4 ^
  --winners example_results/blind_visible_quotient_theta_winners.pkl ^
  --winner-index 0 ^
  --graph-mode frame_random_theta ^
  --max-channel-inputs 128 ^
  --max-channel-backgrounds 128 ^
  --max-state-samples 8192 ^
  --schedule-samples 16 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/observer_visible_period_theta.csv ^
  --plot example_results/fig_observer_visible_period_theta.png
```

Interpretation:

```text
visible_schedule_quotient_max_period = 1
  deterministic visible quotient relaxes to a fixed point.

visible_schedule_quotient_max_period > 1
  first periodic recurrent visible-quotient dynamics signal.

visible_schedule_quotient_recurrent_state_count > 1 with max_period = 1
  multiple fixed points or nondeterministic recurrent structure; inspect cycle lengths.
```
