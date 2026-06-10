# Recursive flux-constraint audit

Adds `relgauge.recursivefluxaudit`, a diagnostic-only module for applying the same finite equalizer/consistency logic one level up, to selected `C2/Z2` flux lattices.

Default mode is algebraic. It loads saved microscopic lattice winners, reads their selected plaquette flux maps, and searches for:

- adjacency equalizers between neighboring plaquette fluxes;
- parity fields on 1-plaquette, adjacent-pair, and 2x2-block patches;
- local vertex-star parity / defect proxies;
- domain-wall and flux-map statistics.

No matter fields, charges, or target defects are added. Defect-like quantities are post-hoc diagnostics.

## Example

```bash
python -m relgauge.recursivefluxaudit example_results/blind_microlattice_winners_q4_3x3.pkl ^
  --mode algebraic ^
  --top-n 50 ^
  --patch-sizes 1,2,4 ^
  --out example_results/recursive_flux_audit_q4_3x3.csv ^
  --plot example_results/fig_recursive_flux_audit_q4_3x3.png
```

For grown 4x4 winners:

```bash
python -m relgauge.recursivefluxaudit example_results/lattice_growth_winners_q4_3to4.pkl ^
  --mode algebraic ^
  --top-n 50 ^
  --patch-sizes 1,2,4 ^
  --out example_results/recursive_flux_audit_q4_4x4.csv ^
  --plot example_results/fig_recursive_flux_audit_q4_4x4.png
```

Optional trajectory diagnostic:

```bash
python -m relgauge.recursivefluxaudit example_results/blind_microlattice_winners_q4_3x3.pkl ^
  --mode trajectory ^
  --top-n 10 ^
  --horizon 16 ^
  --out example_results/recursive_flux_trajectory_q4_3x3.csv
```

Trajectory mode tracks a state-dependent parity readout of selected edge labels. It is a diagnostic, not the primary recursive closure test.
