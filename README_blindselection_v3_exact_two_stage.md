# blindselection v3: exact and two-stage selection

This patch updates `relgauge.blindselection` so blind rule search can use the
strict interface equalizer as the primary selected object.

New CLI options:

```bash
--fitness approx|exact|two_stage
--pretrain-generations N
--exact-tie-breaker EPS
```

Modes:

- `approx`: previous behavior. Uses approximate-equalizer quality as the active
  fitness. Useful for robustness diagnostics, but can have a high random null.
- `exact`: uses the exact zero-error equalizer residual `log_q |S|` as the
  primary fitness. Approximate quality is only a tiny tie-breaker.
- `two_stage`: uses approximate fitness for the first `N` generations, then
  switches to exact `log_q |S|`. Null samples are scored by exact fitness.

Recommended q=4 run:

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
  --out example_results/blind_selection_exact_twostage_q4.csv ^
  --plot example_results/fig_blind_selection_exact_twostage_q4.png
```

Interpretation:

- Strong positive: `null_best_exact_residual_qcoords = 0` and evolved final or
  all-time exact residual is `>0`, ideally `1.0`.
- Approximate-only positives are now treated as pretraining/diagnostics, not as
  final evidence of exact selected law.
- The CSV includes both approximate and exact metrics for audit.
