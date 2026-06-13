# Observer connection automorphism-valid scoring patch (v3)

This patch updates the observer-frame connection selection target from flat path equality to loop automorphism validity.

## Conceptual change

The v2 `blindobserverconnection.py` score rewarded `max_path_agreement`, so it favored flat connections where two completed paths induced the same frame map.  That was too strict for gauge curvature: a valid gauge loop only requires the two path maps to be comparable as a bijective automorphism of the source frame,

```text
Delta = path_1^{-1} path_2 in Aut(F_source),
```

not that `Delta` be the identity.

The v3 score therefore rewards:

```text
soft edge information
+ live exact edge quotients
+ true multi-port observer frames
+ true frame transports
+ branch completion
+ two-branch completion
+ complete branch
+ loop automorphism validity
```

It does **not** reward identity, non-identity, group order, C2, flux, or matter.  Nontrivial holonomy remains a post-hoc audit.

## New metrics

`observerframebundle.py` now reports:

```text
flat_connection
loop_automorphism_valid
loop_automorphism_nontrivial
path_comparability
max_path_comparability
max_loop_automorphism_validity
n_flat_connections
n_loop_automorphism_valid
loop_automorphism_valid_fraction
```

`valid_connection` is retained as the older flat/path-equal diagnostic for compatibility.  The v3 scorer uses `max_loop_automorphism_validity` or global valid holonomy instead.

## Recommended hard run

```bash
python -m relgauge.blindobserverconnection 4 ^
  --graph-mode frame_random_diamond ^
  --seed-mode random ^
  --population 48 ^
  --generations 120 ^
  --elite 6 ^
  --mutation-rate 0.08 ^
  --random-injection 0.10 ^
  --max-state-samples 2048 ^
  --max-channel-inputs 256 ^
  --max-channel-backgrounds 256 ^
  --min-frame-ports 2 ^
  --out example_results/blind_observer_connection_v3_q4.csv ^
  --winners example_results/blind_observer_connection_v3_q4_winners.pkl ^
  --plot example_results/fig_blind_observer_connection_v3_q4.png
```

A clean success may be flat or nontrivial.  The decisive automorphism-valid fields are:

```text
best_max_loop_automorphism_validity
best_valid_holonomy
best_global_valid_holonomy
best_nontrivial_holonomy
best_global_nontrivial_holonomy
```
