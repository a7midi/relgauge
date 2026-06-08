# Approximate Equalizer Patch

Adds:

```text
relgauge/approxequalizer.py
tests/test_approxequalizer.py
```

Run:

```bash
python -m pytest tests/test_approxequalizer.py -q
```

Main experiment:

```bash
python -m relgauge.approxequalizer 3 ^
  --ks-B 2 ^
  --ks-M 2 ^
  --ks-A 2,3 ^
  --ws 1 ^
  --instances 20 ^
  --rule-ensembles random,copy,permutive,block,canalizing ^
  --mutation-rates 0,0.02,0.05,0.1,0.2 ^
  --thresholds 0.15,0.2,0.25 ^
  --out example_results/approx_equalizer.csv ^
  --plot example_results/fig_approx_equalizer.png
```

This module computes an approximate shared interface alphabet by thresholding
weak compatibility edges before forming equalizer components.  It reports both
exact `log_q |S|` and approximate latent information `I(Z_L;Z_R)/log2(q)`.
