# Blind rule-selection patch

This patch adds:

```text
relgauge/blindselection.py
tests/test_blindselection.py
```

It implements a blind search over rule tables for nontrivial approximate
interface equalizers.  The search objective only uses the consistency/equalizer
score from `approxequalizer.py`; copy accuracy, interface mutual information,
and conditional entropy are reported afterward as diagnostics.

## Smoke test

```bash
python -m pytest tests/test_blindselection.py -q
```

## Main run

```bash
python -m relgauge.blindselection 3 ^
  --ks-B 2 ^
  --ks-M 2 ^
  --ks-A 2 ^
  --ws 1 ^
  --null-samples 80 ^
  --runs 5 ^
  --population 16 ^
  --generations 20 ^
  --target interface ^
  --proposal-mode mixed ^
  --entry-rate 0.08 ^
  --table-rate 0.10 ^
  --thresholds 0.45,0.55,0.65 ^
  --out example_results/blind_selection.csv ^
  --plot example_results/fig_blind_selection.png
```

For a stricter full-rule search, use:

```bash
--target all
```

For a faster but more local interface-pipeline search, use the default:

```bash
--target interface
```

## Interpretation

Positive signal:

```text
null_selection_fraction low
evolved_final_best_selection_fraction high
evolved_final_best_fitness > null_best_fitness
post-hoc interface MI and copy accuracy high in winners
```

That means the consistency objective is rediscovering latent-label-preserving
rules without being told what copy, block, canalizing, or permutive rules are.

Negative signal:

```text
evolution does not beat null
selected candidates are isolated / low-MI / mutation-fragile
```

Then the hand-designed structured ensembles are controls rather than evidence of
an emergent selected rule sector.

## Compatibility update

This v2 patch makes `blindselection.py` compatible with both older and newer
`approxequalizer.py` APIs.  If your local `approx_equalizer_measure` does not
accept `min_count` or `require_mutual`, blind selection now automatically omits
those arguments and normalizes the older output column names.
