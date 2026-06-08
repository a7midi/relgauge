# SCDC Diamond Quotient-Cost Patch

This patch adds `relgauge.scdcdiamond`, an experiment that measures the **least local quotient cost** required to turn a diamond consistency condition into a law.

It follows the negative results from the direct phase-channel and random diamond-herald experiments. Instead of asking whether random finite SCCs naturally transmit or correlate schedule phase, this module asks the formal SCDC question:

> How much local alphabet quotienting is required to force the declared diamond consistency constraint?

## Files

Place these files from the patch at your repository root:

```text
relgauge/scdcdiamond.py
tests/test_scdcdiamond.py
README_scdcdiamond_patch.md
```

The module is standalone relative to the earlier new experiments: it only imports `relgauge.core` and standard package dependencies.

## Main command

```bash
python -m relgauge.scdcdiamond 4 ^
  --ks-B 2 ^
  --ks-M 2 ^
  --ks-A 2 ^
  --ws 1 ^
  --instances 20 ^
  --constraint-modes branch_order,sink_equal,strict_schedule ^
  --initial-pool all ^
  --out example_results/scdc_diamond.csv ^
  --plot example_results/fig_scdc_diamond.png
```

On macOS/Linux, replace `^` with `\`.

## Constraint modes

### `branch_order`

Pairs each ordinary full diamond schedule

```text
B,C,D,A
```

with the matching swapped schedule

```text
B,D,C,A
```

using the same internal schedules. For a clean feed-forward diamond, this should cost zero. It is the null/control.

### `sink_equal`

Forces the two sink panels

```text
A[0:w]      receives C
A[w:2w]    receives D
```

to agree modulo a shared local face quotient. This converts the old `sink_equal` herald into a deterministic quotient law and measures the collapse required.

### `strict_schedule`

Forces all admissible schedules to agree modulo quotient. This is intentionally aggressive and often over-collapses. Use it as an upper-bound/control, not as the preferred physical object.

## Main output fields

```text
quotient_cost_qcoords
```

Number of q-ary local coordinates removed by the quotient.

```text
quotient_cost_bits
```

Same cost in bits.

```text
cost_A_qcoords, cost_B_qcoords, cost_C_qcoords, cost_D_qcoords
```

Where the collapse occurred by component.

```text
branch_order_seed_merges
```

Should be zero or near zero in the feed-forward control.

```text
sink_seed_merges
```

Direct sink-panel identifications seeded by the `sink_equal` law.

```text
admissibility_merges
```

Additional identifications forced by local rule-congruence closure.

## Interpretation

A useful positive signal is:

```text
branch_order cost ≈ 0
sink_equal cost small but nonzero
strict_schedule cost large
```

That would mean ordinary branch order is already confluent, but the sink-consistency law has a finite nontrivial quotient cost.

A result like:

```text
sink_equal cost near full A-panel cost
```

means random diamonds can be made consistent only by substantial sink collapse. That supports the current refined theory: raw random SCCs do not naturally create law; SCDC law is a quotient/selection principle.

## Tests

```bash
python -m pytest tests/test_scdcdiamond.py -q
```

In the reconstructed patched tree, the full suite gives:

```text
28 passed
```
