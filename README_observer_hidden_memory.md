# Observer Hidden Memory Experiment

This patch adds a new test for the next D-stage idea: an observer SCC may contain hidden internal memory that absorbs schedule ambiguity while only a coarse visible frame value is exposed at its boundary.

The motivation is the gated section-dynamics result: the theta observer-frame connection can be preserved, but visible coherence-defect dynamics remains schedule-nondeterministic and has no nontrivial recurrent quotient. The hidden-memory experiment asks whether this failure is caused by throwing away the observer's internal SCC state too early.

## New modules

```text
relgauge/observerhiddenmemory.py
relgauge/blindobserverhiddenmemory.py
```

## Main audit

For each sampled microstate `x`, the audit computes:

```text
V(x) = visible observer-frame/coherence output tuple
H(x) = tuple of internal SCC microstate codes for required observers
J(x) = (V(x), H(x))
```

Then, under sampled schedules, it compares:

```text
V -> V'      visible-only transition
J -> V'      hidden-conditioned visible transition
J -> J'      full hidden transition
```

It also builds a color-preserving deterministic quotient of `J`, where the color is the visible output `V`. The quotient is allowed to merge hidden successor states caused by schedule ambiguity, but it is not allowed to merge different visible outputs. This tests whether hidden memory can absorb schedule gauge without erasing visible distinctions.

The audit does not reward or assume motion, nontrivial recurrent classes, C2, flux, particles, localization, or lifetime.

## Run on a selected theta winner

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
  --out example_results/observer_hidden_memory_theta_winner.csv ^
  --plot example_results/fig_observer_hidden_memory_theta_winner.png
```

Important fields:

```text
visible_only_transition_determinism_accuracy
hidden_conditioned_visible_determinism_accuracy
hidden_memory_ambiguity_reduction
memory_quotient_classes
memory_compression_ratio
memory_quotient_source_block_deterministic_fraction
memory_quotient_visible_conflict_source_fraction
memory_quotient_recurrent_classes
memory_quotient_nontrivial_recurrent
schedule_absorption_signal
dynamics_candidate
```

A strong result would look like:

```text
connection_gate_passed = true
hidden_conditioned_visible_determinism_accuracy ~= 1
memory_quotient_source_block_deterministic_fraction ~= 1
memory_quotient_classes > distinct_visible_states
memory_quotient_classes < distinct_joint_states
memory_quotient_nontrivial_recurrent = true   # post hoc, not selected
```

## Blind/seeded hidden-memory selection

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
  --out example_results/blind_hidden_memory_theta.csv ^
  --winners example_results/blind_hidden_memory_theta_winners.pkl ^
  --plot example_results/fig_blind_hidden_memory_theta.png
```

The selector preserves the observer-frame connection gate and rewards only hidden-conditioned visible schedule consistency and compressed memory-quotient determinism. It does not reward recurrent dynamics or motion.

## Focused tests

```bash
PYTHONPATH=. pytest -q \
  tests/test_observerhiddenmemory.py \
  tests/test_blindobserverhiddenmemory.py
```
