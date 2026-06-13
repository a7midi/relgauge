# Blind observer-section dynamics selection

This patch adds:

```text
relgauge/blindobserversectiondynamics.py
```

It is the selection companion to `observersectiondynamics.py` v2.  It targets the next D-stage question:

```text
Can selected local rules make the observer-frame section-obstruction quotient
closed and schedule-consistent, without rewarding motion, lifetime, particles,
nontriviality, flux, C2, or group order?
```

## What is rewarded

The score rewards only consistency prerequisites:

```text
observer-frame connection validity
coherent observer-frame section values
closure of the coherent-section sector under schedules
schedule determinism of coherent residuals
schedule determinism of cycle-syndrome residuals
schedule determinism of minimal-support residual representatives
```

It explicitly does not reward:

```text
nontrivial holonomy
nontrivial recurrent quotient
motion
localized support
particle labels
lifetime
flux values
C2 or any target group
```

Those quantities remain post-hoc diagnostics in the summaries.

## Recommended first run: seeded from a theta connection winner

```bash
python -m relgauge.blindobserversectiondynamics 4 ^
  --seed-winners example_results/blind_theta_connection_q4_winners.pkl ^
  --seed-winner-index 0 ^
  --population 24 ^
  --generations 80 ^
  --elite 4 ^
  --mutation-rate 0.04 ^
  --random-injection 0.05 ^
  --max-channel-inputs 128 ^
  --max-channel-backgrounds 128 ^
  --max-state-samples 2048 ^
  --schedule-samples 16 ^
  --frame-coordinate-mode local_charts ^
  --cycle-objective all ^
  --out example_results/blind_section_dynamics_theta.csv ^
  --winners example_results/blind_section_dynamics_theta_winners.pkl ^
  --plot example_results/fig_blind_section_dynamics_theta.png
```

The decisive fields are:

```text
final_coherent_state_fraction
final_coherent_transition_closed_fraction
final_minimal_support_transition_closed_fraction
final_cycle_syndrome_transition_closed_fraction
final_minimal_support_recurrent_quotient_classes
final_coherent_recurrent_quotient_classes
final_cycle_syndrome_recurrent_quotient_classes
```

A true D-stage success would be post-hoc, not fitness-targeted:

```text
coherent/minimal/syndrome sector closed under schedules
schedule quotient nontrivial on recurrent classes
motion/localization audited after that
```

## Random-start theta run

```bash
python -m relgauge.blindobserversectiondynamics 4 ^
  --graph-mode frame_random_theta ^
  --seed-mode random ^
  --population 32 ^
  --generations 120 ^
  --elite 6 ^
  --mutation-rate 0.06 ^
  --random-injection 0.10 ^
  --max-state-samples 2048 ^
  --schedule-samples 16 ^
  --frame-coordinate-mode local_charts ^
  --cycle-objective all ^
  --out example_results/blind_section_dynamics_random_theta.csv ^
  --winners example_results/blind_section_dynamics_random_theta_winners.pkl
```

This is harder because it must build the connection and the section-closure consistency simultaneously.
