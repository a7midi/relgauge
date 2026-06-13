# Blind observer-section dynamics consistency selection

This patch adds `relgauge/blindobserversectiondynamics.py`, with `relgauge/blindsectiondynamics.py` as a short CLI alias.

The module evolves local rule tables while keeping the graph fixed. Its score rewards only logical consistency requirements:

- live observer-frame connection,
- existence of coherent observer-frame sections,
- closure of the coherent-section subspace under sampled schedules,
- schedule-deterministic transitions of the cohomological/minimal-support residual layers.

It does **not** reward motion, nontrivial recurrent quotient size, holonomy sector, C2, flux, localization, lifetime, particle labels, or matter. These remain post-hoc diagnostics reported by `observersectiondynamics.py` v2.

## Seeded run from a selected theta winner

```bash
python -m relgauge.blindobserversectiondynamics 4 ^
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
  --out example_results/blind_section_dynamics_theta.csv ^
  --winners example_results/blind_section_dynamics_theta_winners.pkl ^
  --plot example_results/fig_blind_section_dynamics_theta.png
```

## Seeded control without a winner pickle

```bash
python -m relgauge.blindobserversectiondynamics 4 ^
  --graph-mode frame_random_theta ^
  --seed-mode theta_twist_c2 ^
  --population 12 ^
  --generations 20 ^
  --elite 3 ^
  --mutation-rate 0.04 ^
  --max-channel-inputs 128 ^
  --max-channel-backgrounds 128 ^
  --max-state-samples 1024 ^
  --schedule-samples 6 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/blind_section_dynamics_seeded.csv ^
  --winners example_results/blind_section_dynamics_seeded_winners.pkl
```

## Key fields

Watch:

- `final_found_autonomous_coherent_residual_quotient`
- `all_time_found_autonomous_coherent_residual_quotient`
- `final_found_nontrivial_recurrent_minimal_support_quotient`
- `all_time_found_nontrivial_recurrent_minimal_support_quotient`
- `best_coherent_state_fraction`
- `best_minimal_support_transition_closed_fraction`
- `best_minimal_support_transition_determinism_accuracy`
- `best_minimal_support_moving_transition_fraction`

A genuine D-stage candidate would require a closed, deterministic coherent/minimal-support residual layer and, post hoc, a nontrivial recurrent quotient with motion/localization diagnostics. Motion and nontriviality are not in the score.
