# Blind observer-section dynamics v2 gate patch

This patch fixes the D-stage optimizer exploit found in the first blind section-dynamics runs.

The earlier selector could obtain an apparently autonomous coherent-residual quotient by destroying the multi-cycle observer-frame connection: live transports collapsed, cycle basis went to zero, and the residual quotient became autonomous only because the gauge arena disappeared.

This patch makes two changes.

## 1. Connection-preservation gate

`relgauge.blindobserversectiondynamics` now has a hard connection gate. When enabled, section-dynamics credit is unavailable unless the selected observer-frame connection is preserved.

Default:

```bash
--connection-gate auto
```

For the theta arena, auto-gate requires:

```text
live edge quotients      >= 6
true multi-port frames   >= 5
true frame transports    >= 6
cycle basis size         >= 2
global chord count       >= 2
global valid holonomies  >= 2
single-port leakage      == 0
full-boundary violation  == 0
```

For double-diamond and diamond modes, corresponding auto targets are inferred from the controlled graph mode. You can override them with:

```bash
--gate-min-live-edge-quotients N
--gate-min-true-frames N
--gate-min-true-transports N
--gate-min-cycle-basis N
--gate-min-global-chords N
--gate-min-global-holonomy N
```

Disable the gate only for debugging:

```bash
--connection-gate off
```

## 2. Observer-level coherence-defect layer

`observersectiondynamics.py` now reports a `coherence_defect` representation. For each required observer, the bit is:

```text
0 = observer has one coherent local frame value
1 = observer is incoherent or unresolved
```

This layer is defined for every microstate and lives on observer nodes, not edge residual representatives. It is the natural candidate for localized section-defect dynamics once the connection is preserved.

New summary fields include:

```text
coherence_defect_patterns
coherence_defect_entropy_bits
coherence_defect_active_fraction
coherence_defect_localized_fraction
coherence_defect_transition_closed_fraction
coherence_defect_transition_determinism_accuracy
coherence_defect_moving_transition_fraction
coherence_defect_schedule_quotient_classes
coherence_defect_recurrent_quotient_classes
coherence_defect_nontrivial_recurrent_quotient
```

## Recommended rerun

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
  --connection-gate auto ^
  --out example_results/blind_section_dynamics_theta_gated.csv ^
  --winners example_results/blind_section_dynamics_theta_gated_winners.pkl ^
  --plot example_results/fig_blind_section_dynamics_theta_gated.png
```

A genuine D-stage candidate now requires:

```text
connection_gate_passed = true
coherence_defect_transition_closed_fraction -> 1
coherence_defect_transition_determinism_accuracy -> 1
```

Post-hoc, look for:

```text
coherence_defect_nontrivial_recurrent_quotient = true
coherence_defect_moving_transition_fraction > 0
```

Those post-hoc fields are not rewarded by the score.
