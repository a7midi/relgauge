# Interface Equalizer Patch

Adds `relgauge.interfaceequalizer`, a relational/equalizer companion to
`relgauge.scdcdiamond`.

The product-local SCDC test showed that forcing sink equality with coordinatewise
quotients collapses the sink observer.  This patch tests the replacement:
represent sink-panel agreement by a shared finite interface alphabet `S` with
maps

```text
f_left(A_left) = f_right(A_right) in S
```

instead of collapsing both panel coordinates to constants.

## Files

Place from the repository root:

```text
relgauge/interfaceequalizer.py
tests/test_interfaceequalizer.py
README_interfaceequalizer_patch.md
```

## Smoke run

```bash
python -m relgauge.interfaceequalizer 4 ^
  --ks-B 2 ^
  --ks-M 2 ^
  --ks-A 2,3 ^
  --ws 1 ^
  --instances 20 ^
  --initial-pool all ^
  --out example_results/interface_equalizer.csv ^
  --plot example_results/fig_interface_equalizer.png
```

## Main diagnostics

* `eq_residual_qcoords`: `log_q |S|`, the shared interface variable left after equalization.
* `eq_cost_qcoords`: `2w - log_q |S|`, the relational cost of turning two panels into a shared interface.
* `product_cost_A_qcoords`: old product-local sink-equality cost on the same instance.
* `overcollapse_avoided_qcoords`: how much less destructive the equalizer is than product-local SCDC.
* `eq_residual_per_width`: near 1 means a full q-ary shared boundary variable survives per boundary vertex; near 0 means the interface still collapses.

## Interpretation

For `w=1`:

* diagonal-like equality gives `eq_residual_qcoords = 1`, `eq_cost_qcoords = 1`.
* complete/noisy mixing gives `eq_residual_qcoords = 0`, `eq_cost_qcoords = 2`.

A strong positive result is:

```text
eq_residual_per_width high
product_cost_A_qcoords grows with kA
eq_cost_qcoords independent of hidden kA
overcollapse_avoided_qcoords positive
```

That means interface consistency is relational/boundary-local rather than sink-trivializing.
