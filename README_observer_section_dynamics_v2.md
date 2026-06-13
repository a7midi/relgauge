# Observer Section Dynamics v2: coherent cohomology audit

This patch upgrades `relgauge/observersectiondynamics.py` from a raw residual-pattern audit to a layered section-obstruction audit.

The earlier version computed port-level residuals

\[
r_e(x)=1[s_{t(e),e}(x)\neq U_e(s_{s(e),e}(x))]
\]

for every live observer-frame transport edge. That was useful, but it mixed three different effects:

1. observer-frame incoherence inside a single observer,
2. genuine edge-gluing failure of a coherent section,
3. gauge/coboundary choices of residual representative.

V2 separates these layers.

## New layers

For each microscopic state `x`, the audit now computes:

- `raw_port`: the old port-level residual vector, even if observer frames are internally incoherent.
- `coherent_edge`: edge residuals only when every required observer has a single coherent frame value.
- `cycle_syndrome`: the Z2 cycle-syndrome vector on a deterministic cycle basis.
- `minimal_support`: a canonical minimal-support residual representative modulo Z2 coboundaries.

The module then computes the schedule-gauge deterministic quotient separately for each layer.

## New key summary fields

Important new summary fields include:

```text
audit_version
cycle_basis_size
cycle_basis_chords
cycle_basis_rows
connection_edge_bits
connection_holonomy_bits
connection_holonomy_weight
coherent_state_count
coherent_state_fraction
coherent_transition_closed_fraction
cycle_syndrome_patterns
cycle_syndrome_schedule_quotient_classes
minimal_support_patterns
minimal_support_mean_size
minimal_support_transition_closed_fraction
minimal_support_nontrivial_recurrent_quotient
layer_summaries
```

A dynamics candidate should not be inferred from raw movement alone. The relevant strict signal is a nontrivial recurrent schedule-invariant quotient in the `minimal_support` or `cycle_syndrome` layer, with the coherent-section sector closed under the sampled schedules.

## Example

```bash
python -m relgauge.observersectiondynamics 4 ^
  --winners example_results/blind_theta_connection_q4_winners.pkl ^
  --winner-index 0 ^
  --frame-coordinate-mode local_charts ^
  --max-state-samples 65536 ^
  --schedule-samples 32 ^
  --out example_results/section_dynamics_theta_winner_v2.csv ^
  --plot example_results/fig_section_dynamics_theta_winner_v2.png
```

Expected conservative interpretation for the current theta winner:

```text
raw residual representatives move,
but coherent sections are rare and not dynamically closed;
cycle syndrome and minimal-support cohomology class are static;
therefore no autonomous section dynamics is established.
```

This is a stricter D-stage audit, not a new fitness function. It does not reward motion, lifetime, localization, particles, nontriviality, or any target group.
