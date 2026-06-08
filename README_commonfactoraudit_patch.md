# Blind winner saving + common-factor audit patch

This patch updates `relgauge.blindselection` and adds `relgauge.commonfactoraudit`.

## Files

Place these files in the repository root:

```text
relgauge/blindselection.py
relgauge/commonfactoraudit.py
tests/test_blindselection.py
tests/test_commonfactoraudit.py
README_commonfactoraudit_patch.md
```

## Save winners during blind selection

Run the two-stage exact q=4 search with winner saving:

```bash
python -m relgauge.blindselection 4 ^
  --ks-B 2 ^
  --ks-M 2 ^
  --ks-A 2 ^
  --ws 1 ^
  --null-samples 500 ^
  --runs 10 ^
  --population 24 ^
  --generations 50 ^
  --fitness two_stage ^
  --pretrain-generations 20 ^
  --target interface ^
  --proposal-mode mixed ^
  --entry-rate 0.08 ^
  --table-rate 0.10 ^
  --thresholds 0.45,0.55,0.65 ^
  --save-winners example_results/blind_winners_q4.pkl ^
  --save-winner-top-n 30 ^
  --out example_results/blind_selection_exact_twostage_q4.csv ^
  --plot example_results/fig_blind_selection_exact_twostage_q4.png
```

The winner file is a local pickle artifact containing cloned `SCDCDiamond` objects and their metrics. Do not load winner pickle files from untrusted sources.

## Audit saved winners

```bash
python -m relgauge.commonfactoraudit example_results/blind_winners_q4.pkl ^
  --top-n 20 ^
  --mutation-rates 0,0.02,0.05,0.1 ^
  --mutation-trials 20 ^
  --out example_results/common_factor_audit_q4.csv ^
  --plot example_results/fig_common_factor_audit_q4.png
```

## Main audit quantities

`exact_residual_qcoords` is `log_q |S|`, the exact shared interface alphabet size.

`source_to_label_mi_over_label_entropy` checks whether the selected label is live: it must depend on the post-B source boundary, not merely exist as a formal compatibility partition.

`one_step_label_persistence_norm` checks whether the selected label persists into another tick.

`mutation_survival_at_max_rate` checks whether exact selection survives the largest requested mutation rate.

## Tests

```bash
python -m pytest tests/test_blindselection.py tests/test_commonfactoraudit.py -q
```

The reconstructed package passed the full suite with this patch installed.
