# Diamond phase-consistency patch

Adds `relgauge/diamond.py` and `tests/test_diamond.py`.

The chain experiment `phasechannel.py` tested whether a finite SCC observer can
retain another SCC's schedule phase.  This module tests a different mechanism:
whether a downstream consistency predicate in a diamond

```text
        B
       / \
      C   D
       \ /
        A
```

selects or correlates otherwise independent branch phases.

The measured object is

```text
P(s = 1 | Phi_C, Phi_D)
```

where `s` is a finite herald/consistency event.  The default `sink_equal` herald
checks whether two paired panels in the sink observer A agree after the diamond
has evolved.  Conditioning on `s=1`, the module reports

```text
I(Phi_C ; Phi_D | s=1)
```

plus herald efficiency.  A nonzero mutual information with reasonable herald
efficiency is evidence for a diamond phase-consistency relation.  A high value
at tiny efficiency should be treated as post-selection-like.

## Files

Place from repo root:

```text
relgauge/diamond.py
README_diamond_patch.md
tests/test_diamond.py
```

## Smoke test

```bash
python -m pytest tests/test_diamond.py -q
```

## Typical run

```bash
python -m relgauge.diamond 4 ^
  --ks-B 2 --ks-C 2 --ks-D 2 --ks-A 2 ^
  --ws 1 ^
  --horizons 1,2,3 ^
  --instances 20 ^
  --phase-modes cut,erased_value ^
  --protocols first ^
  --herald-modes sink_equal ^
  --out example_results/diamond.csv ^
  --plot example_results/fig_diamond.png
```

For a control that repeatedly drives the same branch phase, add:

```bash
--protocols first,same
```

For a direct branch agreement herald rather than sink-panel agreement, add:

```bash
--herald-modes sink_equal,branch_equal
```

For faster exploratory sweeps, especially at q=4, add:

```bash
--no-robust
```

This skips possible/certain set-propagation and keeps the stochastic herald
matrix, which is the main result.

## Main columns

- `herald_efficiency`: average P(s=1) under a uniform phase-pair prior.
- `relation_mi_bits`: I(Phi_C; Phi_D | s=1).
- `relation_mi_norm`: the same MI normalized by the smaller branch phase entropy.
- `selection_bits`: KL posterior phase-pair distribution to the uniform prior.
- `marginal_selection_C_bits`, `marginal_selection_D_bits`: phase filtering of each branch separately.
- `robust_possible_fraction`: fraction of phase pairs for which s=1 is possible under nondeterministic schedules.
- `robust_certain_fraction`: fraction of phase pairs for which s=1 is certain under nondeterministic schedules.
- `herald_matrix`: JSON matrix for P(s=1 | Phi_C, Phi_D).

## Interpretation

- `DIAMOND PHASE-CONSISTENCY`: downstream consistency correlates branch phases.
- `MARGINAL PHASE FILTERING`: consistency selects phases independently but does not relate branches.
- `NO GENERIC DIAMOND PHASE CONSTRAINT`: random diamonds do not create a robust phase relation.
- `LOW-EFFICIENCY / POST-SELECTION-LIKE`: possible relation, but probably a selection artifact unless efficiency can be improved.
