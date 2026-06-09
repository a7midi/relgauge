# Temporal Chain Patch

Adds `relgauge.temporalchain`, a two-diamond transport/composition experiment:

```text
        B0
       /  \
     C1    D1
       \  /
        A1
       /  \
     C2    D2
       \  /
        A2
```

The module measures the exact equalizer label `S1` at `A1`, the exact equalizer
label `S2` at `A2`, and whether `S1` is transported to `S2`.

Key metrics:

- `s1_residual_qcoords`, `s2_residual_qcoords`: exact shared alphabets at the
  two convergence points.
- `transport_mi_over_Hs2`: normalized `I(S1; S2) / H(S2)`.
- `source_to_s2_mi_over_Hs2`: normalized `I(source; S2) / H(S2)`.
- `s1_to_s2_best_accuracy`: best deterministic map accuracy from `S1` to `S2`.
- `live_transport_selected`: exact labels exist, `S1` predicts `S2`, and the
  final label is source-live.

Recommended command:

```bash
python -m relgauge.temporalchain 4 ^
  --ks-B 2 ^
  --ks-M 2 ^
  --ks-A 2 ^
  --ws 1 ^
  --instances 20 ^
  --rule-ensembles random,copy,permutive,block,canalizing,constant ^
  --mutation-rates 0,0.02,0.05,0.1 ^
  --initial-mode joint_random ^
  --n-random-initial 4096 ^
  --out example_results/temporal_chain.csv ^
  --plot example_results/fig_temporal_chain.png
```

`joint_random` is recommended for null-calibrated tests because it varies
downstream initial memory as noise. `source_all` is useful for debugging and
for measuring an ideal source-to-chain map, but can produce accidental random
transport in tiny systems.
