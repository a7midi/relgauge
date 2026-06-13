# Observer hidden memory v2: visible quotienting

This patch adds the visible-quotienting correction to the hidden-memory D-stage audit.

The previous hidden-memory audit asked whether raw observer-internal SCC state could make visible observer-frame dynamics schedule-deterministic. It found useful ambiguity reduction, but the color-preserving memory quotient usually stayed almost as large as the sampled joint state space and did not absorb schedule ambiguity.

The v2 correction tests the cleaner consistency question:

```text
Find the finest nontrivial quotient of visible + hidden observer state whose transition is schedule-independent.
```

This allows raw visible distinctions to be identified when they are schedule-gauge. It does not insert motion, matter, particles, curvature, C2, or any target group.

## New quotient layers

`observerhiddenmemory.py` now reports four quotient families in the `_quotient.csv` output:

```text
color_preserving_memory
visible_schedule
joint_schedule_visible_quotient
induced_visible_from_joint_schedule
```

The new layers are:

- `visible_schedule`: deterministic quotient of visible states alone.
- `joint_schedule_visible_quotient`: deterministic quotient of joint `(visible, hidden)` states, allowed to merge different visible outputs.
- `induced_visible_from_joint_schedule`: the visible partition induced by the joint schedule quotient.

## Important summary fields

```text
visible_schedule_quotient_signal
joint_schedule_quotient_signal
visible_quotient_schedule_absorption_signal
visible_schedule_quotient_classes
visible_schedule_quotient_retention_fraction
visible_schedule_quotient_source_block_deterministic_fraction
joint_schedule_quotient_classes
joint_schedule_quotient_retention_fraction
joint_schedule_quotient_compression_fraction
joint_schedule_quotient_source_block_deterministic_fraction
induced_visible_quotient_classes
induced_visible_quotient_retention_fraction
induced_visible_quotient_transition_determinism_accuracy
```

A strong D-stage signal would have the connection gate passed, a non-collapsed visible or joint schedule quotient, deterministic quotient transitions, and a nontrivial recurrent quotient only as a post-hoc diagnostic.

## Audit command

```bash
python -m relgauge.observerhiddenmemory 4 ^
  --winners example_results/blind_theta_connection_q4_winners.pkl ^
  --winner-index 0 ^
  --graph-mode frame_random_theta ^
  --max-channel-inputs 128 ^
  --max-channel-backgrounds 128 ^
  --max-state-samples 4096 ^
  --schedule-samples 8 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/observer_visible_quotient_theta.csv ^
  --plot example_results/fig_observer_visible_quotient_theta.png
```

## Selection command

```bash
python -m relgauge.blindobserverhiddenmemory 4 ^
  --seed-winners example_results/blind_theta_connection_q4_winners.pkl ^
  --seed-winner-index 0 ^
  --population 24 ^
  --generations 60 ^
  --elite 4 ^
  --mutation-rate 0.04 ^
  --random-injection 0.05 ^
  --max-channel-inputs 128 ^
  --max-channel-backgrounds 128 ^
  --max-state-samples 2048 ^
  --schedule-samples 8 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/blind_visible_quotient_theta.csv ^
  --winners example_results/blind_visible_quotient_theta_winners.pkl ^
  --plot example_results/fig_blind_visible_quotient_theta.png
```
