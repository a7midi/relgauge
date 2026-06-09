# Attractor audit patch

Adds `relgauge.attractoraudit`, a recurrent-orbit audit for saved blind-selection winners.

The tool reconstructs the selected exact interface alphabet `S`, freezes its left/right label maps,
then iterates the full finite diamond dynamics under one or more deterministic admissible schedules.
It reports whether `S` is constant on recurrent attractors, cycles as a phase variable, or is merely a
formal instantaneous equalizer.

Example:

```bash
python -m relgauge.attractoraudit example_results/blind_winners_q4_full.pkl ^
  --top-n 50 ^
  --schedule-mode canonical ^
  --initial-pool all ^
  --observer-init zero ^
  --out example_results/attractor_audit_q4_full.csv ^
  --plot example_results/fig_attractor_audit_q4_full.png
```

Schedule-sensitive check:

```bash
python -m relgauge.attractoraudit example_results/blind_winners_q4_full.pkl ^
  --top-n 10 ^
  --schedule-mode all ^
  --max-schedules 0 ^
  --out example_results/attractor_audit_q4_schedules.csv
```

Key fields:

- `constant_basin_fraction`: fraction of all states whose attractor has a constant valid `S` label.
- `stable_label_entropy_norm`: whether different basins carry different constant labels.
- `phase_basin_fraction`: fraction of states whose attractor carries a nonconstant periodic `S` sequence.
- `attractor_label_transition_mi_norm`: normalized `I(S_t; S_{t+1})` on recurrent cycles.

Verdicts:

- `ATTRACTOR-CONSERVED S`: selected labels are stable basin invariants.
- `PHASE-CYCLING S`: selected labels persist as recurrent phase variables.
- `FORMAL / NONPERSISTENT S`: selected equalizers do not become recurrent invariants.
